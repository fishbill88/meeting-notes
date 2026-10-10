/**
 * Renders text with `terms` highlighted, or plain when there are none.
 *
 * Ported from the mobile component. The highlight is amber-on-tint plus a
 * bottom border, not colour alone — colour-only emphasis fails both WCAG and
 * anyone reading a printed note.
 *
 * Today only the consultation template supplies terms, so meeting notes
 * render unhighlighted.
 */

import { memo } from "react";

import { highlightSpans } from "../lib/medicalTerms";

interface Props {
  text: string;
  terms?: string[];
}

function HighlightedTextInner({ text, terms = [] }: Props) {
  const spans = highlightSpans(text, terms);

  return (
    <>
      {spans.map((span, index) =>
        span.highlighted ? (
          <mark key={index} className="term">
            {span.text}
          </mark>
        ) : (
          <span key={index}>{span.text}</span>
        ),
      )}
    </>
  );
}

export const HighlightedText = memo(HighlightedTextInner);
