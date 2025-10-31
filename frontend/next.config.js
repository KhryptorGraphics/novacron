/** @type {import('next').NextConfig} */
const nextConfig = {
  // Experimental features
  experimental: {
    // App router is now default in Next.js 13.5+
  },

  // Compiler optimizations
  compiler: {
    // Remove console statements in production
    removeConsole: process.env.NODE_ENV === 'production',
  },

  // Image optimization
  images: {
    remotePatterns: [
      {
        protocol: 'https',
        hostname: '**',
      },
    ],
  },

  // Basic redirects
  async redirects() {
    return [
      {
        source: '/login',
        destination: '/auth/login',
        permanent: true,
      },
      {
        source: '/register',
        destination: '/auth/register',
        permanent: true,
      },
    ];
  },

  // Basic optimizations
  poweredByHeader: false,
  reactStrictMode: true,
  swcMinify: false, // Disable SWC to avoid crashes
  trailingSlash: false,

  // ESLint configuration
  eslint: {
    // Warning: This allows production builds to complete even with ESLint errors
    ignoreDuringBuilds: true,
  },

  // TypeScript configuration
  typescript: {
    // Warning: This allows production builds to complete even with TypeScript errors
    ignoreBuildErrors: true,
  },
};

module.exports = nextConfig;