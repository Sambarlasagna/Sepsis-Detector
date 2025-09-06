import { query } from "./db";

export async function getUserFromGoogleId(googleId) {
  const res = await query("SELECT * FROM users WHERE google_id = $1", [googleId]);
  return res.rows[0] || null;
}

export async function createUser(googleId, name, picture = null, email = null) {
  const res = await query(
    "INSERT INTO users (google_id, name, picture, email) VALUES ($1, $2, $3, $4) RETURNING *",
    [googleId, name, picture, email]
  );
  return res.rows[0];
}
