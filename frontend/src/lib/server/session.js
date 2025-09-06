import { randomBytes } from "crypto";

const sessions = new Map(); // in-memory store for demo

export async function generateSessionToken() {
  return randomBytes(32).toString("hex");
}

export async function createSession(token, userId) {
  sessions.set(token, { userId, createdAt: Date.now() });
}

export async function getSession(token) {
  return sessions.get(token) || null;
}

export async function setSessionTokenCookie(cookieStore, token) {
  cookieStore.set({
    name: process.env.SESSION_COOKIE_NAME,
    value: token,
    path: "/",
    httpOnly: true,
  });
}
