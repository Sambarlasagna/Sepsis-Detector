// app/page.js
import { redirect } from 'next/navigation';

export default function Home() {
  // This will redirect immediately to /dashboard
  redirect('/dashboard');
}
