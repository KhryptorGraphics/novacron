import type { Metadata } from "next";
import "./globals.css";

// Import components
import { ThemeProvider } from "@/components/theme-provider";
import { Toaster } from "@/components/ui/toaster";
import { QueryProvider } from "@/providers/query-provider";
import { AuthProvider } from "@/hooks/useAuth";
import { RBACProvider } from "@/contexts/RBACContext";
import { ErrorBoundary } from "@/components/error-boundary";

// Metadata for the application
export const metadata: Metadata = {
  title: "NovaCron - Distributed Cloud Hypervisor",
  description: "Next-generation distributed cloud hypervisor with advanced monitoring and management",
};

// PRODUCTION FIX: Force all routes to be dynamic (no static generation)
// This bypasses SSR errors for urgent production launch
export const dynamic = 'force-dynamic';
export const dynamicParams = true;
export const revalidate = 0;

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="font-sans antialiased">
        <ErrorBoundary>
          <AuthProvider>
            <RBACProvider>
              <QueryProvider>
                <ThemeProvider attribute="class" defaultTheme="system" enableSystem>
                  {children}
                  <Toaster />
                </ThemeProvider>
              </QueryProvider>
            </RBACProvider>
          </AuthProvider>
        </ErrorBoundary>
      </body>
    </html>
  );
}
