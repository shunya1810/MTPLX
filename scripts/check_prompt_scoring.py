"""HTTP-only scoring gate for an already running, externally guarded server.

Gemma scoring had no successful parent output before #551. This checks
repeatability and a bounded change across admitted widths, not parent class A.
"""

import argparse
import json
import math
from pathlib import Path
from urllib.request import Request, urlopen


def checked_scores(response, *, width):
    assert response["mtplx_stats"]["prefill_chunk_tokens"] == width, "admitted width"
    scores = response["choices"][0]["logprobs"]
    count = response["usage"]["prompt_tokens"]
    assert count > 1
    assert len(scores["token_ids"]) == len(scores["token_logprobs"]) == count
    assert len(scores["top_logprobs"]) == count
    assert scores["token_logprobs"][0] is None
    assert all(math.isfinite(value) for value in scores["token_logprobs"][1:]), "finite scores"
    assert all(
        math.isfinite(value)
        for row in scores["top_logprobs"] for value in row.values()
    ), "finite top scores"
    return scores


def check_repeat(first, repeated, *, width):
    scores = checked_scores(first, width=width)
    again = checked_scores(repeated, width=width)
    assert scores == again, "repeated scoring changed"
    return scores


def compare(left, right, *, bound=0.005):
    assert math.isfinite(bound) and 0 <= bound <= 0.005, "filed absolute bound"
    assert left["head"] == right["head"] and left["pack"] == right["pack"], "same tree and pack"
    assert left["width"] != right["width"], "two different admitted widths"
    assert set(left["cases"]) == set(right["cases"]) == {"short", "long"}
    result = {"class": "C", "absolute_logprob_bound": bound, "cases": {}}
    for label, a in left["cases"].items():
        b = right["cases"][label]
        x = check_repeat(a["first"], a["repeated"], width=left["width"])
        y = check_repeat(b["first"], b["repeated"], width=right["width"])
        assert x["token_ids"] == y["token_ids"], "same tokenized prompt"
        if label == "long":
            assert len(x["token_ids"]) > max(2048, left["width"], right["width"])
        deltas = [abs(p - q) for p, q in zip(
            x["token_logprobs"][1:], y["token_logprobs"][1:], strict=True
        )]
        agreements = []
        for p, q in zip(x["top_logprobs"][1:], y["top_logprobs"][1:], strict=True):
            assert p and q
            deltas.extend(abs(p[token] - q[token]) for token in p.keys() & q.keys())
            agreements.append(max(p, key=p.get) == max(q, key=q.get))
        peak = max(deltas)
        top1 = sum(agreements) / len(agreements)
        assert peak <= bound, f"{label}: absolute logprob delta {peak} > {bound}"
        assert top1 >= 0.985, f"{label}: top-1 agreement {top1} < 0.985"
        result["cases"][label] = {
            "compared_values": len(deltas), "max_abs_logprob_delta": peak,
            "top1_text_agreement": top1, "exactly_equal_across_widths": x == y,
            "repeated_exactly": True,
        }
    result["scope"] = (
        "Scored tokens and shared top-K entries; top-1 compares decoded text. "
        "Not full-distribution KL or parent class A."
    )
    return result


def capture(args):
    receipt = {"head": args.head, "pack": args.pack, "width": args.width, "cases": {}}

    def request(path, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = Request(args.base_url + path, data=data, headers={
            "Content-Type": "application/json", "x-mtplx-cache-mode": "bypass",
        })
        with urlopen(req, timeout=240) as response:
            return json.load(response)

    receipt["health"] = request("/health")
    for label, prompt in (
        ("short", "The current status is ready."),
        ("long", "This is a synthetic prompt for checking local scoring.\n" * 350),
    ):
        body = {"prompt": prompt, "echo": True, "logprobs": 5, "max_tokens": 0}
        case = receipt["cases"][label] = {"request": body}
        for name in ("first", "repeated"):
            case[name] = request("/v1/completions", body)
            Path(args.output).write_text(json.dumps(receipt, indent=2) + "\n")
        scores = check_repeat(case["first"], case["repeated"], width=args.width)
        if label == "long":
            assert len(scores["token_ids"]) > max(2048, args.width)
    print(json.dumps({"finite": True, "repeated_exactly": True, "width": args.width}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    record = modes.add_parser("capture")
    record.add_argument("--base-url", default="http://127.0.0.1:18391")
    record.add_argument("--width", type=int, required=True)
    record.add_argument("--head", required=True)
    record.add_argument("--pack", required=True)
    record.add_argument("--output", required=True)
    check = modes.add_parser("compare")
    check.add_argument("left")
    check.add_argument("right")
    check.add_argument("--bound", type=float, default=0.005)
    check.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.mode == "capture":
        capture(args)
    else:
        result = compare(
            json.loads(Path(args.left).read_text()),
            json.loads(Path(args.right).read_text()), bound=args.bound,
        )
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))


if __name__ == "__main__":
    main()
