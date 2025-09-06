import { google } from "googleapis";

const oauth2Client = new google.auth.OAuth2(
  process.env.GOOGLE_CLIENT_ID,
  process.env.GOOGLE_CLIENT_SECRET,
  `${process.env.NEXT_PUBLIC_BASE_URL}/login/google/callback`
);

export const googleAuthUrl = oauth2Client.generateAuthUrl({
  access_type: "offline",
  scope: ["openid", "profile", "email"],
  prompt: "consent",
});

export async function validateAuthorizationCode(code) {
  const { tokens } = await oauth2Client.getToken(code);
  oauth2Client.setCredentials(tokens);
  return {
    idToken: () => tokens.id_token,
  };
}
