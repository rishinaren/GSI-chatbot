import { useId } from "react";
import { ANSWER_LEVELS, answerLevel } from "../answerLevels";

// Three stops on one track. The native range input does the work (keyboard,
// screen readers, dragging); the track, fill and stop dots drawn behind it only
// show where it is.
export default function AnswerLevelSlider({ value, onChange }) {
  const labelId = useId();
  const current = answerLevel(value);
  const position = ANSWER_LEVELS.indexOf(current);

  return (
    <div className="level-field">
      <div className="level-field-head">
        <span className="level-field-label" id={labelId}>Answer level</span>
        <span className="level-field-value" aria-live="polite">{current.label}</span>
      </div>

      <div className="level-slider" style={{ "--level-pos": position / (ANSWER_LEVELS.length - 1) }}>
        <div className="level-track" aria-hidden="true">
          <span className="level-fill" />
          {ANSWER_LEVELS.map((level, index) => (
            <span
              key={level.value}
              className={`level-stop ${index <= position ? "passed" : ""}`}
              style={{ "--stop-pos": index / (ANSWER_LEVELS.length - 1) }}
            />
          ))}
        </div>
        <input
          type="range"
          className="level-range"
          min={0}
          max={ANSWER_LEVELS.length - 1}
          step={1}
          value={position}
          onChange={(event) => onChange(ANSWER_LEVELS[Number(event.target.value)].value)}
          aria-labelledby={labelId}
          aria-valuetext={current.label}
        />
      </div>

      <div className="level-ticks">
        {ANSWER_LEVELS.map((level) => (
          <button
            key={level.value}
            type="button"
            className={level.value === current.value ? "on" : ""}
            onClick={() => onChange(level.value)}
            tabIndex={-1}
          >
            {level.label}
          </button>
        ))}
      </div>
      <p className="level-hint">{current.hint}</p>
    </div>
  );
}
