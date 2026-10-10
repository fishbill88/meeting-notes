import { memo } from 'react';
import { StyleSheet, Text, TextStyle } from 'react-native';
import { highlightSpans } from '../lib/medicalTerms';

/**
 * Renders text with medical terms (lexicon + extra LLM-flagged terms)
 * highlighted in amber — readable on the dark slate background.
 */
function HighlightedTextInner({
  text,
  extraTerms = [],
  style,
}: {
  text: string;
  extraTerms?: string[];
  style?: TextStyle | TextStyle[];
}) {
  const spans = highlightSpans(text, extraTerms);
  return (
    <Text style={style}>
      {spans.map((span, i) =>
        span.highlighted ? (
          <Text key={i} style={styles.highlight}>{span.text}</Text>
        ) : (
          <Text key={i}>{span.text}</Text>
        ),
      )}
    </Text>
  );
}

export const HighlightedText = memo(HighlightedTextInner);

const styles = StyleSheet.create({
  highlight: {
    backgroundColor: 'rgba(250, 204, 21, 0.22)',
    color: '#FDE047',
    fontWeight: '600',
  },
});
