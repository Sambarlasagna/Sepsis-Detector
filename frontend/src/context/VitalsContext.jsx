"use client";
import { createContext, useContext, useState } from "react";

const VitalsContext = createContext(null);

export const VitalsProvider = ({ children }) => {
  // { patientId: currentIndex }
  const [vitalsProgress, setVitalsProgress] = useState({});

  const updateVitalsProgress = (patientId, index) => {
    setVitalsProgress((prev) => ({
      ...prev,
      [patientId]: index,
    }));
  };

  return (
    <VitalsContext.Provider value={{ vitalsProgress, updateVitalsProgress }}>
      {children}
    </VitalsContext.Provider>
  );
};

export const useVitalsContext = () => useContext(VitalsContext);