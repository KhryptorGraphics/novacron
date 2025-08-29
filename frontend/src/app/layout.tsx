import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";

// Import components
import { ThemeProvider } from "@/components/theme-provider";
import { Toaster } from "@/components/ui/toaster";
import { QueryProvider } from "@/providers/query-provider";

// Font optimization for NovaCron
const inter = Inter({
  subsets: ["latin"],
  variable: "--font-sans",
  display: "swap",
  preload: true,
  fallback: [
    "-apple-system",
    "BlinkMacSystemFont", 
    "Segoe UI",
    "Roboto",
    "Helvetica Neue",
    "Arial",
    "sans-serif"
  ],
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono", 
  display: "swap",
  preload: false, // Load on demand for code
  fallback: [
    "SF Mono",
    "Monaco",
    "Cascadia Code", 
    "Roboto Mono",
    "Consolas",
    "Courier New",
    "monospace"
  ],
});

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
      <body className={`${inter.variable} ${jetbrainsMono.variable} font-sans`}>
        <QueryProvider>
          <ThemeProvider attribute="class" defaultTheme="system" enableSystem>
            {children}
            <Toaster />
          </ThemeProvider>
        </QueryProvider>
      </body>
    </html>
  );
}
