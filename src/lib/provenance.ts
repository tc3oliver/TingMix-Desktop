/** The host labels placeholder output "Mock — not ASR"; anything else names the model and its status. */
export const isMock = (label: string | null | undefined): boolean => !label || label.startsWith("Mock");

export const MOCK_NOTE = "Mock — not real ASR: these words are placeholders, not recognized speech.";

/** One line saying which backend produced a transcript, written into exports so the file carries it too. */
export function exportNote(label: string | null | undefined): string | undefined {
  return isMock(label) ? MOCK_NOTE : undefined;
}
