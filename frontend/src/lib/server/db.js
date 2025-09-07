import { neon } from '@neondatabase/serverless';

const sql = neon(`${process.env.DATABASE_URL}`);

export async function execute(query, params = []) {
    return await sql.query(query, params);
}

export async function fetch(query, params = []) {
    return await sql.query(query, params);
}

export async function fetchOne(query, params = []) {
    let data = await fetch(query, params);
    return data[0];
}