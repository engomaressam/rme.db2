"use client";

export default function Logout() {
  const handleLogout = () => {
    localStorage.removeItem("auth_token"); // Remove token
    window.location.href = "/login"; // Redirect
  };

  return (
    <button
      onClick={handleLogout}
      className="bg-red-500 text-white p-2 rounded"
    >
      Logout
    </button>
  );
}
