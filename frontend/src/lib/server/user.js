import { execute, fetch, fetchOne } from "./db";

export async function createUser(googleId, name, picture) {
    const count = await fetchOne(`SELECT COUNT(*) FROM "user"`); 
    const newId = count.count + 1;
    const row = await fetchOne(`INSERT INTO "user" (id, google_id, name, picture) VALUES ($1, $2, $3, $4);`,
        [newId, googleId, name, picture]);
    // if (!row) {
    //     throw new Error("Unexpected error");
    // }
    const user = {
        id: newId,
        googleId,
        name,
        picture
    };
    return user;
}

export async function getUserFromGoogleId(googleId) {
    const row = await fetchOne(`SELECT id, google_id, name, picture FROM "user" WHERE google_id = $1`, [googleId]);
    if (!row) {
        return null;
    }
    const user = {
        id: row.id,
        googleId: row.google_id,
        name: row.name,
        picture: row.picture
    };
    return user;
}