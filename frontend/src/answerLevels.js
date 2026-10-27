// How much an answer explains. The same sources back every level; what moves is
// how much is explained and how much detail is given. The values are the names the
// API and saved chats use, so they stay put when the labels change.
export const ANSWER_LEVELS = [
  { value: "beginner", label: "Succinct", hint: "A brief summary of the key points." },
  { value: "intermediate", label: "Balanced", hint: "Key information with relevant context and explanations." },
  { value: "advanced", label: "Detailed", hint: "In-depth explanations with technical details." },
];

export const DEFAULT_ANSWER_LEVEL = "intermediate";

export function answerLevel(value) {
  return ANSWER_LEVELS.find((level) => level.value === value) ?? ANSWER_LEVELS[1];
}
