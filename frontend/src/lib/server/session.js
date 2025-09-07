import { randomBytes } from "crypto";

// In-memory store (for demo only, won't persist across requests)
const sessions = new Map();

export async function generateSessionToken() {
  return randomBytes(32).toString("hex");
}

export async function createSession(token, userId) {
  sessions.set(token, { userId, createdAt: Date.now() });
}

export async function getSession(token) {
  return sessions.get(token) || null;
}

// ✅ cookieStore must be awaited and set with correct object syntax
export async function setSessionTokenCookie(cookieStore, token) {
  await cookieStore.set({
    name: process.env.SESSION_COOKIE_NAME || "session_token",
    value: token,
    path: "/",
    httpOnly: true,
    sameSite: "lax",
    secure: process.env.NODE_ENV === "production",
  });
}
