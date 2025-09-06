import { NextResponse } from "next/server";
import { googleAuthUrl } from "../../../lib/server/oauth";
import { cookies } from "next/headers";
import { randomBytes } from "crypto";

export async function GET(req) {
  const url = new URL(req.url);
  const redirect = url.searchParams.get("redirect") || "/";

  const state = randomBytes(16).toString("hex");

  const cookieStore = cookies();
  cookieStore.set("google_oauth_state", state, { path: "/" });
  cookieStore.set("post_login_redirect", redirect, { path: "/" });

  return NextResponse.redirect(googleAuthUrl + `&state=${state}`);
}
