import { Redirect } from 'expo-router';

export const SETUP_DONE_KEY = 'setup_done_v1';

export default function SetupScreen() {
  return <Redirect href="/settings" />;
}
