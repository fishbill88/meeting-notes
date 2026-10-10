import { Tabs, useRouter } from 'expo-router';
import { useEffect } from 'react';
import { ColorValue, Text } from 'react-native';
import { allModelsReady } from '../lib/models';

function TabIcon({ emoji, color }: { emoji: string; color: ColorValue }) {
  return <Text style={{ fontSize: 20, color: color as string }}>{emoji}</Text>;
}

export default function RootLayout() {
  const router = useRouter();

  useEffect(() => {
    allModelsReady().then((ready) => {
      if (!ready) router.replace('/settings');
    });
  }, []);

  return (
    <Tabs
      screenOptions={{
        headerStyle: { backgroundColor: '#1E3A5F' },
        headerTintColor: '#fff',
        headerTitleStyle: { fontWeight: '700' },
        tabBarActiveTintColor: '#3B82F6',
        tabBarInactiveTintColor: '#94A3B8',
        tabBarStyle: { backgroundColor: '#0F172A', borderTopColor: '#1E293B' },
      }}
    >
      <Tabs.Screen
        name="index"
        options={{
          title: 'Meeting Notes AI',
          tabBarLabel: 'Record',
          tabBarIcon: ({ color }) => <TabIcon emoji="🎙" color={color} />,
        }}
      />
      <Tabs.Screen
        name="history"
        options={{
          title: 'Meetings',
          tabBarLabel: 'Meetings',
          tabBarIcon: ({ color }) => <TabIcon emoji="📋" color={color} />,
        }}
      />
      <Tabs.Screen
        name="settings"
        options={{
          title: 'Settings',
          tabBarLabel: 'Settings',
          tabBarIcon: ({ color }) => <TabIcon emoji="⚙️" color={color} />,
        }}
      />
      <Tabs.Screen name="setup" options={{ href: null }} />
    </Tabs>
  );
}
