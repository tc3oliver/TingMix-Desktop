import { describe, expect, it } from "vitest";
import { translator } from "../i18n";
import { gigabytes } from "../features/setup/SetupScreen";
import { labelText } from "./provenance";

const t = translator("en");

describe("labels shown to users", () => {
  it("name the model without its status word or engine", () => {
    expect(labelText("tingmix-large-mlx-fp32-r75071a1 (mlx) — EXPERIMENTAL", t)).toBe("TingMix Large");
    expect(labelText("Mock — not ASR", t)).toBe("Demo mode");
  });
  it("show download sizes in decimal gigabytes", () => {
    const num = (n: number, o?: Intl.NumberFormatOptions) => new Intl.NumberFormat("en", o).format(n);
    expect(gigabytes(1_308_067_132 + 113_932_911, num)).toBe("1.4 GB");
    expect(gigabytes(500_000_000, num)).toBe("500 MB");
  });
});
