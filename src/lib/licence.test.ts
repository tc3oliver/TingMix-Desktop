import { describe, expect, it } from "vitest";
import { translator } from "../i18n";
import { licenceName } from "./licence";

describe("licence names", () => {
  it("use the localized wording for known ids and the manifest name otherwise", () => {
    const l = { id: "tingmix-large-model", name: "x" };
    expect(licenceName(l, translator("en"))).toBe("TingMix-Large model terms");
    expect(licenceName(l, translator("zh-TW"))).toBe("TingMix-Large 模型使用條款");
    expect(licenceName({ id: "google-embeddinggemma-2", name: "x" }, translator("zh-TW"))).toBe("Google EmbeddingGemma 2 授權與禁止用途政策");
    expect(licenceName({ id: "other", name: "Other Terms" }, translator("zh-TW"))).toBe("Other Terms");
  });
});
