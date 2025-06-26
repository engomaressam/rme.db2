"use client";

import {
  createContext,
  useContext,
  useEffect,
  useState,
  useRef,
  useMemo,
} from "react";
import { useRouter } from "next/navigation";

interface User {
  token: string;
  name?: string;
  email?: string;
}

interface AuthContextType {
  user: User | null;
  login: (token: string, name?: string, email?: string) => void;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const [user, setUser] = useState<User | null>(null);
  const router = useRouter();
  const initialized = useRef(false); // ✅ Prevents unnecessary useEffect runs

  useEffect(() => {
    if (initialized.current) return;
    initialized.current = true; // ✅ Ensures this runs only ONCE

    const token = sessionStorage.getItem("auth_token");
    const name = sessionStorage.getItem("auth_name");
    const email = sessionStorage.getItem("auth_email");

    if (token) {
      setUser({ token, name: name || "", email: email || "" });
    } else {
      router.replace("/login"); // ✅ Redirect only if user is NOT logged in
    }
  }, []); // ✅ Runs ONLY once

  const login = (token: string, name?: string, email?: string) => {
    sessionStorage.setItem("auth_token", token);
    if (name) sessionStorage.setItem("auth_name", name);
    if (email) sessionStorage.setItem("auth_email", email);

    setUser({ token, name, email });
    router.push("/"); // ✅ Redirect to home or dashboard
  };

  const logout = () => {
    sessionStorage.removeItem("auth_token");
    sessionStorage.removeItem("auth_name");
    sessionStorage.removeItem("auth_email");

    setUser(null);
    router.push("/login"); // ✅ Redirect after logout
  };

  // ✅ MEMOIZE CONTEXT VALUE TO PREVENT UNNECESSARY RERENDERS
  const authContextValue = useMemo(() => ({ user, login, logout }), [user]);

  return (
    <AuthContext.Provider value={authContextValue}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
};
