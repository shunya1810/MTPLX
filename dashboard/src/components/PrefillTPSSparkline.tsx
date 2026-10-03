import { useEffect, useMemo, useRef } from "react";
import uPlot from "uplot";
import { Card } from "./Card";
import { usePrefillHistory } from "../hooks/usePolling";
import { fmtTokS } from "../lib/utils";
import { autoAxisSize } from "../lib/uplotAxis";

export function PrefillTPSSparkline() {
  const { data } = usePrefillHistory();
  const wrapperRef = useRef<HTMLDivElement | null>(null);
  const plotRef = useRef<uPlot | null>(null);

  const { aligned, mean } = useMemo(() => {
    const xs: number[] = [];
    const ys: number[] = [];
    const rows = data?.history ?? [];
    let sum = 0;
    let count = 0;
    rows.forEach((row) => {
      if (typeof row.prefill_tok_s !== "number") return;
      xs.push(row.t);
      ys.push(row.prefill_tok_s);
      sum += row.prefill_tok_s;
      count += 1;
    });
    return {
      aligned: [xs, ys] as uPlot.AlignedData,
      mean: count > 0 ? sum / count : null,
    };
  }, [data]);

  useEffect(() => {
    const wrapper = wrapperRef.current;
    if (!wrapper) return;
    const opts: uPlot.Options = {
      width: wrapper.clientWidth,
      height: 140,
      padding: [4, 8, 4, 0],
      cursor: { drag: { x: false, y: false, setScale: false } },
      scales: {
        x: { time: true },
        y: { range: (_self, mn, mx) => [Math.max(0, mn * 0.85), mx * 1.1] },
      },
      axes: [
        { stroke: "rgba(200,210,220,0.4)", show: true, gap: 4, size: 22 },
        {
          stroke: "rgba(200,210,220,0.4)",
          values: (_self, ticks) => ticks.map((t) => `${t.toFixed(0)}`),
          size: autoAxisSize,
        },
      ],
      legend: { show: false },
      series: [
        {},
        {
          stroke: "rgba(79,182,243,0.95)",
          width: 1.6,
          fill: "rgba(79,182,243,0.15)",
          points: { show: false },
          paths: uPlot.paths.spline?.(),
        },
      ],
    };
    const plot = new uPlot(opts, aligned, wrapper);
    plotRef.current = plot;
    const onResize = () => plot.setSize({ width: wrapper.clientWidth, height: 140 });
    window.addEventListener("resize", onResize);
    return () => {
      window.removeEventListener("resize", onResize);
      plot.destroy();
      plotRef.current = null;
    };
  }, []);

  useEffect(() => {
    plotRef.current?.setData(aligned);
  }, [aligned]);

  return (
    <Card
      title="Prefill tok/s · last 100"
      subtitle={mean !== null ? `mean ${fmtTokS(mean)} tok/s` : "no prefill samples yet"}
    >
      <div ref={wrapperRef} className="w-full" />
    </Card>
  );
}
