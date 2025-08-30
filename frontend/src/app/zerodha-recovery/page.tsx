import dynamic from 'next/dynamic';

// Dynamically import to avoid SSR issues
const ZerodhaRecoveryDashboard = dynamic(
  () => import('@/components/zerodha/ZerodhaRecoveryDashboard'),
  { ssr: false }
);

export default function ZerodhaRecoveryPage() {
  return <ZerodhaRecoveryDashboard />;
}
