import { NextRequest, NextResponse } from "next/server";

export function middleware(req: NextRequest) {
  console.log("Middleware: ", req.nextUrl.pathname);
  const token =
    req.cookies.get("auth_token")?.value ||
    req.headers.get("authorization")?.replace("Bearer ", "");

  const protectedRoutes = new Set([
    "/",
    "/dashboard",
    "/profile",
    "/settings",
    "/add-expenditure",
    "/expenditures",
  ]);

  const publicRoutes = new Set(["/login", "/signup", "/public"]);

  const pathname = req.nextUrl.pathname;

  // Redirect unauthenticated users to `/login` for all protected routes
  if (!token && protectedRoutes.has(pathname) && pathname !== "/login") {
    return NextResponse.redirect(new URL("/login", req.url));
  }

  // Prevent logged-in users from accessing `/login` or `/signup`
  if (token && publicRoutes.has(pathname)) {
    return NextResponse.redirect(new URL("/dashboard", req.url));
  }

  return NextResponse.next();
}

// Middleware applies to all pages
export const config = {
  matcher: "/:path*",
};
