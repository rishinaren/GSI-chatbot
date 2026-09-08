const PAGE_MARGIN = 54;
const PAGE_BOTTOM = 54;
const BODY_SIZE = 10.5;
const BODY_LINE_HEIGHT = 15;
const REPORT_FONT_FILE = "DejaVuSans.ttf";
const REPORT_FONT_NAME = "DejaVuSans";
let reportFontPromise = null;

function arrayBufferToBase64(buffer) {
  const bytes = new Uint8Array(buffer);
  const chunks = [];
  const chunkSize = 0x8000;
  for (let offset = 0; offset < bytes.length; offset += chunkSize) {
    chunks.push(String.fromCharCode(...bytes.subarray(offset, offset + chunkSize)));
  }
  return btoa(chunks.join(""));
}

async function loadReportFont() {
  if (!reportFontPromise) {
    reportFontPromise = fetch("/fonts/DejaVuSans.ttf").then(async (response) => {
      if (!response.ok) {
        throw new Error("The report font could not be loaded.");
      }
      return arrayBufferToBase64(await response.arrayBuffer());
    });
  }
  try {
    return await reportFontPromise;
  } catch (error) {
    // A transient asset failure should not make every later export fail too.
    reportFontPromise = null;
    throw error;
  }
}

function isAbortError(error) {
  return error?.name === "AbortError";
}

export function reportFileName(question, scope = "answer") {
  if (scope === "conversation") {
    return "GSI-conversation-report.pdf";
  }

  const subject = String(question ?? "")
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 54)
    .replace(/-+$/g, "");

  return `GSI-report${subject ? `-${subject}` : ""}.pdf`;
}

// The answer is stored as Markdown. A PDF should read like the rendered answer,
// not expose the punctuation used to format it in the browser.
export function markdownToReportText(markdown) {
  return String(markdown ?? "")
    .replace(/\r\n?/g, "\n")
    .replace(/```[^\n]*\n?([\s\S]*?)```/g, "$1")
    .replace(/!\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, "$1 ($2)")
    .replace(/^\s{0,3}#{1,6}\s+/gm, "")
    .replace(/^\s*>\s?/gm, "")
    .replace(/^\s*[-*+]\s+/gm, "- ")
    .replace(/\*\*([^*]+)\*\*/g, "$1")
    .replace(/__([^_]+)__/g, "$1")
    .replace(/~~([^~]+)~~/g, "$1")
    .replace(/`([^`]+)`/g, "$1")
    .replace(/\\([*_`~])/g, "$1")
    .replace(/[ \t]+$/gm, "")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

export function formatReportCitation(citation, index) {
  const attached = citation?.source_kind === "attachment";
  const source = attached
    ? "Your document"
    : citation?.standard_id || citation?.title || `Source ${index + 1}`;
  const details = [];

  if (citation?.title && citation.title !== source) {
    details.push(citation.title);
  }
  if (citation?.section) {
    details.push(`Section ${citation.section}`);
  }
  if (citation?.page_start != null) {
    const pages =
      citation.page_end != null && citation.page_end !== citation.page_start
        ? `pages ${citation.page_start}-${citation.page_end}`
        : `page ${citation.page_start}`;
    details.push(pages);
  }

  const sourceUrl = !attached ? citation?.source_url : null;
  const citationText = `[${index + 1}] ${source}${details.length ? `, ${details.join(", ")}` : ""}`;
  return sourceUrl ? `${citationText}\n${sourceUrl}` : citationText;
}

function addPageNumber(pdf, pageNumber, pageCount) {
  const width = pdf.internal.pageSize.getWidth();
  const height = pdf.internal.pageSize.getHeight();
  pdf.setDrawColor(222, 225, 228);
  pdf.setLineWidth(0.5);
  pdf.line(PAGE_MARGIN, height - 37, width - PAGE_MARGIN, height - 37);
  pdf.setFont(REPORT_FONT_NAME, "normal");
  pdf.setFontSize(8);
  pdf.setTextColor(114, 120, 127);
  pdf.text(`Page ${pageNumber} of ${pageCount}`, width - PAGE_MARGIN, height - 22, {
    align: "right",
  });
}

export async function createReportPdf(
  { question, answer, citations = [], exchanges, scope = "answer" },
  { fontBase64 } = {},
) {
  const { jsPDF } = await import("jspdf");
  const pdf = new jsPDF({
    orientation: "portrait",
    unit: "pt",
    format: "letter",
    compress: true,
    putOnlyUsedFonts: true,
  });
  const reportFont = fontBase64 || (await loadReportFont());
  pdf.addFileToVFS(REPORT_FONT_FILE, reportFont);
  pdf.addFont(REPORT_FONT_FILE, REPORT_FONT_NAME, "normal", "Identity-H");

  const pageWidth = pdf.internal.pageSize.getWidth();
  const pageHeight = pdf.internal.pageSize.getHeight();
  const contentWidth = pageWidth - PAGE_MARGIN * 2;
  let y = PAGE_MARGIN;

  function startPage() {
    pdf.setFillColor(15, 45, 79);
    pdf.rect(0, 0, pageWidth, 7, "F");
    y = PAGE_MARGIN;
  }

  function nextPage() {
    pdf.addPage();
    startPage();
  }

  function ensureRoom(height) {
    if (y + height > pageHeight - PAGE_BOTTOM) {
      nextPage();
    }
  }

  function drawHeading(text, { first = false } = {}) {
    // Non-title headings reserve the first body line too, so a heading is never
    // stranded at the foot of a page.
    ensureRoom(first ? 42 : 55);
    if (!first) y += 18;
    pdf.setFont("helvetica", "bold");
    pdf.setFontSize(first ? 17 : 11);
    pdf.setTextColor(first ? 15 : 92, first ? 45 : 99, first ? 79 : 106);
    pdf.text(text, PAGE_MARGIN, y);
    y += first ? 25 : 19;
  }

  function drawText(text, { indent = 0, lineHeight = BODY_LINE_HEIGHT } = {}) {
    pdf.setFont(REPORT_FONT_NAME, "normal");
    pdf.setFontSize(BODY_SIZE);
    pdf.setTextColor(34, 39, 44);

    const sourceLines = String(text || "").split("\n");
    sourceLines.forEach((sourceLine) => {
      if (!sourceLine.trim()) {
        ensureRoom(lineHeight * 0.6);
        y += lineHeight * 0.6;
        return;
      }

      const listItem = sourceLine.match(/^(\s*)(- |\d+[.)]\s+)(.*)$/);
      const prefix = listItem?.[2] ?? "";
      const body = listItem?.[3] ?? sourceLine;
      const itemIndent = listItem ? 15 : indent;
      const lines = pdf.splitTextToSize(body, contentWidth - itemIndent);

      lines.forEach((line, lineIndex) => {
        ensureRoom(lineHeight);
        if (prefix && lineIndex === 0) {
          pdf.text(prefix.trim(), PAGE_MARGIN + indent, y);
        }
        pdf.text(line, PAGE_MARGIN + itemIndent, y);
        y += lineHeight;
      });
    });
  }

  const reportExchanges =
    exchanges?.length > 0 ? exchanges : [{ question, answer, citations }];
  const isConversation = scope === "conversation";

  startPage();
  drawHeading(isConversation ? "GSI CONVERSATION REPORT" : "GSI RESPONSE REPORT", { first: true });
  pdf.setDrawColor(196, 164, 105);
  pdf.setLineWidth(1.4);
  pdf.line(PAGE_MARGIN, y - 10, PAGE_MARGIN + 52, y - 10);
  y += 7;

  reportExchanges.forEach((exchange, exchangeIndex) => {
    if (exchangeIndex > 0) {
      if (y + 82 > pageHeight - PAGE_BOTTOM) {
        nextPage();
      } else {
        y += 22;
        pdf.setDrawColor(222, 225, 228);
        pdf.setLineWidth(0.7);
        pdf.line(PAGE_MARGIN, y, pageWidth - PAGE_MARGIN, y);
        y += 4;
      }
    }

    const suffix = isConversation ? ` ${exchangeIndex + 1}` : "";
    drawHeading(isConversation ? `USER QUESTION${suffix}` : "1. USER QUESTION");
    drawText(markdownToReportText(exchange.question) || "No question was provided.");

    drawHeading(isConversation ? `CHATBOT ANSWER${suffix}` : "2. CHATBOT ANSWER");
    drawText(markdownToReportText(exchange.answer) || "No answer was provided.");

    drawHeading("CITATIONS");
    const exchangeCitations = exchange.citations ?? [];
    if (exchangeCitations.length) {
      exchangeCitations.forEach((citation, citationIndex) => {
        if (citationIndex > 0) y += 6;
        drawText(formatReportCitation(citation, citationIndex), {
          indent: 0,
          lineHeight: 14,
        });
      });
    } else {
      drawText("None provided.");
    }
  });

  const pageCount = pdf.getNumberOfPages();
  for (let pageNumber = 1; pageNumber <= pageCount; pageNumber += 1) {
    pdf.setPage(pageNumber);
    addPageNumber(pdf, pageNumber, pageCount);
  }

  return pdf.output("blob");
}

function downloadBlob(blob, fileName) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = fileName;
  link.style.display = "none";
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export async function saveReportPdf(report) {
  const firstQuestion = report.question ?? report.exchanges?.[0]?.question;
  const fileName = reportFileName(firstQuestion, report.scope);
  let fileHandle = null;

  // Chromium exposes the OS save panel directly. Ask for the destination first
  // so the chooser appears immediately while the click still has user activation.
  if (typeof window.showSaveFilePicker === "function") {
    try {
      fileHandle = await window.showSaveFilePicker({
        suggestedName: fileName,
        types: [
          {
            description: "PDF document",
            accept: { "application/pdf": [".pdf"] },
          },
        ],
        excludeAcceptAllOption: true,
      });
    } catch (error) {
      if (isAbortError(error)) return { canceled: true };
      throw error;
    }
  }

  const blob = await createReportPdf(report);
  if (fileHandle) {
    const writable = await fileHandle.createWritable();
    try {
      await writable.write(blob);
      await writable.close();
    } catch (error) {
      await writable.abort().catch(() => {});
      throw error;
    }
  } else {
    // Safari and Firefox do not consistently expose a programmatic save panel;
    // a PDF download is their native, user-configurable fallback.
    downloadBlob(blob, fileName);
  }

  return { canceled: false, fileName };
}
