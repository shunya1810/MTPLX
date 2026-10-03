import type uPlot from "uplot";

type SizedAxis = Omit<uPlot.Axis, "font"> & {
  _size?: number;
  font?: string | [string, number, number];
};

/**
 * uPlot sizes an axis at a fixed 50px by default, which clips wider tick
 * labels such as "52 tok/s" or "1250". Size the axis to its longest label
 * instead, using the font and pixel scale uPlot uses to draw the ticks.
 */
export function autoAxisSize(
  self: uPlot,
  values: string[] | null,
  axisIdx: number,
  cycleNum: number,
): number {
  const axis = self.axes[axisIdx] as SizedAxis;
  // uPlot re-runs sizing until it converges; keep the first result.
  if (cycleNum > 1) return axis._size ?? 50;
  let size = (axis.ticks?.size ?? 10) + (axis.gap ?? 5);
  const previousFont = self.ctx.font;
  const font = Array.isArray(axis.font) ? axis.font[0] : axis.font;
  if (font) self.ctx.font = font;
  let width = 0;
  for (const value of values ?? []) {
    width = Math.max(width, self.ctx.measureText(value).width);
  }
  self.ctx.font = previousFont;
  size += width / devicePixelRatio;
  return Math.ceil(size);
}
