// How much an answer explains. The same sources back every level; what moves is
// how much is explained and how much detail is given.
export const ANSWER_LEVELS = [
  { value: "beginner", label: "Beginner", hint: "Plain language. Terms explained as they come up." },
  { value: "intermediate", label: "Intermediate", hint: "The key details, for working engineers." },
  { value: "advanced", label: "Advanced", hint: "Full specifics: conditions, equations, limits." },
];

export const DEFAULT_ANSWER_LEVEL = "intermediate";

export function answerLevel(value) {
  return ANSWER_LEVELS.find((level) => level.value === value) ?? ANSWER_LEVELS[1];
}
