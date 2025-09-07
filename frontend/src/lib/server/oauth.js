import { Google } from "arctic";

export const google = new Google(
	process.env.GOOGLE_CLIENT_ID ?? "",
	process.env.GOOGLE_CLIENT_SECRET ?? "",
	"https://sepsis-sense.vercel.app/login/google/callback",
  "http://localhost:3000/login/google/callback" 
);