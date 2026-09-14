import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import {
  createReportPdf,
  formatReportCitation,
  markdownToReportText,
  reportFileName,
  saveReportPdf,
} from "./reportPdf.js";

test("builds safe, descriptive file names for either export scope", () => {
  assert.equal(
    reportFileName("What is ASTM D5321 / D6243?"),
    "GSI-report-what-is-astm-d5321-d6243.pdf",
  );
  assert.equal(reportFileName("Ignored", "conversation"), "GSI-conversation-report.pdf");
});

test("turns stored Markdown into readable report text", () => {
  assert.equal(
    markdownToReportText("## Result\n\n**Peak strength** is [documented](https://example.com)."),
    "Result\n\nPeak strength is documented (https://example.com).",
  );
});

test("formats library and attachment citations without leaking attachment links", () => {
  assert.equal(
    formatReportCitation(
      {
        standard_id: "ASTM D5321",
        title: "Direct Shear Test Method",
        section: "8.2",
        page_start: 3,
        source_url: "https://example.com/d5321",
      },
      0,
    ),
    "[1] ASTM D5321, Direct Shear Test Method, Section 8.2, page 3\nhttps://example.com/d5321",
  );
  assert.equal(
    formatReportCitation(
      {
        source_kind: "attachment",
        title: "field-report.pdf",
        page_start: 7,
        source_url: "https://example.com/private",
      },
      1,
    ),
    "[2] Your document, field-report.pdf, page 7",
  );
  assert.equal(
    formatReportCitation(
      {
        source_kind: "design_guidance",
        standard_id: "DWG-6E-V2",
        title: "Designing with Geosynthetics, 6th Edition, Volume 2",
        section: "5.3",
        page_start: 566,
      },
      2,
    ),
    "[3] Design guidance, Designing with Geosynthetics, 6th Edition, Volume 2, Section 5.3, page 566",
  );
});

test("creates a PDF with Unicode technical notation", async () => {
  const fontBase64 = fs.readFileSync(
    new URL("../public/fonts/DejaVuSans.ttf", import.meta.url),
  ).toString("base64");
  const blob = await createReportPdf(
    {
      question: "Is σ ≥ 10 kN/m²?",
      answer: "Yes — use μ = 0.5.",
      citations: [],
    },
    { fontBase64 },
  );
  const bytes = Buffer.from(await blob.arrayBuffer());

  assert.equal(blob.type, "application/pdf");
  assert.equal(bytes.subarray(0, 5).toString("ascii"), "%PDF-");
  assert.ok(bytes.length > 10_000);
});

test("opens the native save picker before generating the PDF", async () => {
  const originalWindow = globalThis.window;
  const originalFetch = globalThis.fetch;
  const events = [];
  let writtenBlob = null;

  globalThis.window = {
    showSaveFilePicker: async (options) => {
      events.push("picker");
      assert.match(options.suggestedName, /^GSI-report-/);
      return {
        createWritable: async () => ({
          write: async (blob) => {
            events.push("write");
            writtenBlob = blob;
          },
          close: async () => events.push("close"),
          abort: async () => {},
        }),
      };
    },
  };
  globalThis.fetch = async () => {
    events.push("font");
    return new Response(fs.readFileSync(new URL("../public/fonts/DejaVuSans.ttf", import.meta.url)));
  };

  try {
    const result = await saveReportPdf({
      question: "How is shear strength reported?",
      answer: "Use the cited test result.",
      citations: [],
    });

    assert.equal(result.canceled, false);
    assert.deepEqual(events, ["picker", "font", "write", "close"]);
    assert.equal(writtenBlob?.type, "application/pdf");
  } finally {
    if (originalWindow === undefined) {
      delete globalThis.window;
    } else {
      globalThis.window = originalWindow;
    }
    globalThis.fetch = originalFetch;
  }
});
