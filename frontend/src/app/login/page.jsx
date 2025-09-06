"use client";
import React from "react";
import { useSearchParams } from "next/navigation";

export default function LoginPage() {
  const searchParams = useSearchParams();
  const redirectTo = searchParams.get("redirect") || "/";

  const handleGoogleLogin = () => {
    window.location.href = `/login/google?redirect=${encodeURIComponent(redirectTo)}`;
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100">
      <h1 className="text-4xl font-bold text-green-700 mb-6">Login</h1>
      <p className="text-gray-600 mb-8 text-center max-w-md">
        Sign in to continue to your dashboard or patients page.
      </p>
      <button
        onClick={handleGoogleLogin}
        className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-lg"
      >
        Sign in with Google
      </button>
    </div>
  );
}
