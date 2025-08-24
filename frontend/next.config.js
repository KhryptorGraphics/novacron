/** @type {import('next').NextConfig} */
const nextConfig = {
  // Enable experimental features for performance
  experimental: {
    // App directory is stable in Next.js 13.4+
    appDir: true,
    // Enable server components optimization
    serverComponentsExternalPackages: [],
  },

  // Compiler optimizations
  compiler: {
    // Remove console statements in production
    removeConsole: process.env.NODE_ENV === 'production',
  },

  // Performance and bundle optimizations
  webpack: (config, { buildId, dev, isServer, defaultLoaders, webpack }) => {
    // Optimize bundle splitting
    if (!dev && !isServer) {
      config.optimization.splitChunks = {
        chunks: 'all',
        cacheGroups: {
          default: false,
          vendors: false,
          // Vendor chunk for React and core libraries
          vendor: {
            name: 'vendor',
            chunks: 'all',
            test: /[\\/]node_modules[\\/](react|react-dom|next)[\\/]/,
            priority: 20,
          },
          // UI components chunk
          ui: {
            name: 'ui',
            chunks: 'all',
            test: /[\\/]node_modules[\\/](@radix-ui|lucide-react|tailwindcss)[\\/]/,
            priority: 15,
          },
          // Charts and visualization chunk
          visualization: {
            name: 'visualization',
            chunks: 'all',
            test: /[\\/]node_modules[\\/](chart\.js|react-chartjs-2|d3|framer-motion)[\\/]/,
            priority: 15,
          },
          // Data fetching chunk
          data: {
            name: 'data',
            chunks: 'all',
            test: /[\\/]node_modules[\\/](@tanstack|react-use-websocket|zod)[\\/]/,
            priority: 10,
          },
          // Common chunk for shared modules
          common: {
            name: 'common',
            chunks: 'all',
            minChunks: 2,
            priority: 5,
            reuseExistingChunk: true,
          },
        },
      };
    }

    // Optimize for real-time applications
    config.resolve.alias = {
      ...config.resolve.alias,
      // Optimize bundle size for production
      ...(process.env.NODE_ENV === 'production' && {
        'react-dom/client': 'react-dom/client',
      }),
    };

    return config;
  },

  // Image optimization
  images: {
    // Enable modern image formats
    formats: ['image/webp', 'image/avif'],
    // Configure domains for external images
    remotePatterns: [
      {
        protocol: 'https',
        hostname: '**',
      },
    ],
    // Optimize for dashboard images
    deviceSizes: [640, 768, 1024, 1280, 1600],
    imageSizes: [16, 32, 48, 64, 96, 128, 256, 384],
  },

  // Headers for performance and security
  async headers() {
    return [
      {
        // Apply to all routes
        source: '/(.*)',
        headers: [
          // Security headers
          {
            key: 'X-Frame-Options',
            value: 'DENY',
          },
          {
            key: 'X-Content-Type-Options',
            value: 'nosniff',
          },
          {
            key: 'Referrer-Policy',
            value: 'origin-when-cross-origin',
          },
          // Performance headers
          {
            key: 'X-DNS-Prefetch-Control',
            value: 'on',
          },
        ],
      },
      {
        // Static assets caching
        source: '/static/(.*)',
        headers: [
          {
            key: 'Cache-Control',
            value: 'public, max-age=31536000, immutable',
          },
        ],
      },
    ];
  },

  // Redirects for clean URLs
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

  // Environment variables for client-side
  env: {
    CUSTOM_KEY: process.env.CUSTOM_KEY,
  },

  // Enable gzip compression
  compress: true,

  // Generate ETags for caching
  generateEtags: true,

  // Power optimizations for dashboard
  poweredByHeader: false,

  // React strict mode for development
  reactStrictMode: true,

  // SWC minification (faster than Terser)
  swcMinify: true,

  // Trailing slash consistency
  trailingSlash: false,

  // TypeScript configuration
  typescript: {
    // Skip type checking during build if you have a separate CI step
    // ignoreBuildErrors: false,
  },

  // ESLint configuration
  eslint: {
    // Run ESLint during build
    // ignoreDuringBuilds: false,
  },

  // Output configuration for deployment
  output: process.env.NODE_ENV === 'production' ? 'standalone' : undefined,
};

module.exports = nextConfig;