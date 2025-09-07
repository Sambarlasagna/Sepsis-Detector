import Papa from "papaparse";

/**
 * Loads CSV data for a given patient.
 *
 * @param {string} id - Patient ID (e.g., "patient1", "patient2").
 * @returns {Promise<Array<Object>>} - Parsed CSV rows.
 */
export default function loadPatientData(id) {
  return new Promise((resolve, reject) => {
    // We'll assume the files are inside /public/data/patient1.csv, patient2.csv, etc.
    const csvPath = `/${id}.csv`;

    Papa.parse(csvPath, {
      header: true, // Parse with column headers
      download: true, // Fetch from public folder
      dynamicTyping: true, // Convert numbers automatically
      complete: (result) => {
        // Filter out any completely empty rows (PapaParse sometimes adds an empty row at the end)
        const filtered = result.data.filter((row) => Object.keys(row).length > 1);
        resolve(filtered);
      },
      error: (error) => {
        console.error(`❌ Failed to load CSV for ${id}:`, error);
        reject(error);
      },
    });
  });
}