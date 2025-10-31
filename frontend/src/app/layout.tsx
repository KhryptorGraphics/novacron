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
