import { NativeEventEmitter, NativeModules, PermissionsAndroid, Platform } from 'react-native';
import * as FileSystem from 'expo-file-system/legacy';

const { WavRecorder } = NativeModules;

/** Matches the native module: 16 kHz mono 16-bit PCM. */
export const RECORDER_SAMPLE_RATE = 16000;

const emitter = new NativeEventEmitter(WavRecorder);

async function requestMicPermission(): Promise<void> {
  if (Platform.OS !== 'android') return;
  const granted = await PermissionsAndroid.request(
    PermissionsAndroid.PERMISSIONS.RECORD_AUDIO,
    {
      title: 'Microphone Permission',
      message: 'Meeting Notes AI needs microphone access to record meetings.',
      buttonPositive: 'Allow',
    },
  );
  if (granted !== PermissionsAndroid.RESULTS.GRANTED) {
    throw new Error('Microphone permission denied');
  }
}

function wavPath(): string {
  const base = (FileSystem.cacheDirectory ?? 'file:///').replace(/\/$/, '');
  return base + '/recording.wav';
}

/**
 * Start mic capture to a WAV file. When `emitChunks` is true the native
 * module also streams base64 PCM buffers to JS (see subscribeAudioChunks)
 * so live transcription can run off the same mic session.
 */
export async function startWavRecording(emitChunks = false): Promise<void> {
  await requestMicPermission();
  const path = wavPath().replace(/^file:\/\//, '');
  await WavRecorder.startRecording(path, emitChunks);
}

export async function stopWavRecording(): Promise<string> {
  const path: string = await WavRecorder.stopRecording();
  return path;
}

/** Subscribe to live base64 PCM chunks (16-bit LE mono @16kHz). Returns unsubscribe. */
export function subscribeAudioChunks(cb: (chunkBase64: string) => void): () => void {
  const sub = emitter.addListener('WavRecorderChunk', cb);
  return () => sub.remove();
}
