// app/logout/route.js

import { getCurrentSession, invalidateSession, deleteSessionTokenCookie } from "@/lib/server/session";
import { cookies } from "next/headers";
import { NextResponse } from "next/server";
export async function GET() {
	const { session } = await getCurrentSession();
	if (session) {
		const cookieStore = cookies();
		await invalidateSession(session.id);
		await deleteSessionTokenCookie(cookieStore);
	}
	// Always redirect to login (even if not logged in)
	return NextResponse.redirect(new URL("/", process.env.NEXT_PUBLIC_SITE_URL || "http://localhost:3000"));
}
