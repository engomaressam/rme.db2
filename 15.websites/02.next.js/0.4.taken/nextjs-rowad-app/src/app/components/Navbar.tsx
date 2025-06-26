"use client";

import { useAuth } from "../utils/AuthProvider";
import {
  Navbar,
  NavbarBrand,
  NavbarContent,
  NavbarItem,
  Button,
  Link,
  Dropdown,
  DropdownTrigger,
  DropdownMenu,
  DropdownItem,
  Avatar,
} from "@heroui/react";
import { useRouter } from "next/navigation";
import Image from "next/image";
import { useState, useEffect } from "react";
import { MenuIcon } from "lucide-react";

export const RMELogo = () => {
  return (
    <Image
      src="/rowad-AI-logo.png"
      alt="Rowad AI Logo"
      width={150}
      height={75}
      priority
    />
  );
};

import React, { memo } from "react";

export default memo(function NavbarComponent() {
  const { user, logout } = useAuth() as {
    user: { name: string; email: string; avatar?: string } | null;
    logout: () => void;
  };
  const router = useRouter();
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [isClient, setIsClient] = useState(false); // Prevent SSR mismatch

  // Ensure client-side rendering before accessing user data
  useEffect(() => {
    setIsClient(true);
  }, []);

  if (!isClient) return null; // Prevents hydration mismatch
  console.log(user);
  // Extract first two characters of the username or default to "U"
  const userInitials =
    user?.name && user.name.trim() ? user.name.slice(0, 2).toUpperCase() : "U";

  return (
    <Navbar shouldHideOnScroll>
      <NavbarBrand>
        <RMELogo />
      </NavbarBrand>

      {/* Mobile Menu Toggle Button */}
      <NavbarContent className="sm:hidden">
        <button
          onClick={() => setIsMenuOpen(!isMenuOpen)}
          aria-label="Toggle Menu"
          aria-expanded={isMenuOpen}
        >
          <MenuIcon size={24} />
        </button>
      </NavbarContent>

      {/* Navbar Links (Only Visible After Login) */}
      {user && (
        <NavbarContent
          className={`${
            isMenuOpen ? "flex" : "hidden"
          } sm:flex gap-4 justify-center`}
        >
          <NavbarItem>
            <Link color="foreground" href="/">
              Home
            </Link>
          </NavbarItem>
          <NavbarItem>
            <Link color="foreground" href="/add-expenditure">
              Add Expenditure
            </Link>
          </NavbarItem>
          <Dropdown>
            <DropdownTrigger>
              <NavbarItem style={{ cursor: "pointer" }}>Reports</NavbarItem>
              {/* <NavbarItem>
            <Link color="foreground" href="/expenditures">
              Expenditures
            </Link>
          </NavbarItem> */}
            </DropdownTrigger>
            <DropdownMenu>
              <DropdownItem
                key="expenditures"
                onPress={() => router.push("/expenditures")}
              >
                Expenditures
              </DropdownItem>
              <DropdownItem
                key="invoices_collection"
                onPress={() => router.push("/invoices_collection")}
              >
                Invoices & Collection
              </DropdownItem>
            </DropdownMenu>
          </Dropdown>
        </NavbarContent>
      )}

      {/* Authentication & User Dropdown */}
      <NavbarContent justify="end">
        {user ? (
          <Dropdown>
            <DropdownTrigger>
              {user.avatar ? (
                <Avatar
                  src={user.avatar}
                  alt="User Avatar"
                  className="cursor-pointer"
                />
              ) : (
                <div className="w-10 h-10 flex items-center justify-center bg-gray-300 text-gray-700 rounded-full cursor-pointer">
                  {userInitials}
                </div>
              )}
            </DropdownTrigger>
            <DropdownMenu>
              <DropdownItem
                key="profile"
                onPress={() => router.push("/profile")}
              >
                Profile
              </DropdownItem>
              <DropdownItem
                key="settings"
                onPress={() => router.push("/settings")}
              >
                Settings
              </DropdownItem>
              <DropdownItem key="logout" color="danger" onPress={logout}>
                Logout
              </DropdownItem>
            </DropdownMenu>
          </Dropdown>
        ) : (
          <NavbarItem>
            <Button color="secondary" onPress={() => router.push("/signup")}>
              Sign Up
            </Button>
          </NavbarItem>
        )}
      </NavbarContent>
    </Navbar>
  );
});
