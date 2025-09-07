import { cookies } from "next/headers";
import jwt from "jsonwebtoken";
import { validateAuthorizationCode } from "../../../../lib/server/oauth";
import { generateSessionToken, createSession, setSessionTokenCookie } from "../../../../lib/server/session";
import { createUser, getUserFromGoogleId } from "../../../../lib/server/user";

export async function GET(req) {
  const url = new URL(req.url);
  const code = url.searchParams.get("code");
  const state = url.searchParams.get("state");

  const cookieStore = cookies();
  const storedState = cookieStore.get("google_oauth_state")?.value;

  if (!code || !state || !storedState || state !== storedState) {
    return new Response("Invalid OAuth state", { status: 400 });
  }

  let tokens;
  try {
    tokens = await validateAuthorizationCode(code);
  } catch (e) {
    console.error("OAuth error", e);
    return new Response("OAuth error", { status: 400 });
  }

  const decoded = jwt.decode(tokens.idToken());
  const googleUserId = decoded.sub;
  const username = decoded.name;
  const picture = decoded.picture || null;
  const email = decoded.email || null;

  const user =
    (await getUserFromGoogleId(googleUserId)) ||
    (await createUser(googleUserId, username, picture, email));

  const sessionToken = await generateSessionToken();
  await createSession(sessionToken, user.id);
  await setSessionTokenCookie(cookieStore, sessionToken);

  const redirectTo = cookieStore.get("post_login_redirect")?.value || "/";
  return new Response(null, { status: 302, headers: { Location: redirectTo } });
}