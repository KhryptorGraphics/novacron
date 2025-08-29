#!/usr/bin/env node

/**
 * Storage Optimization Command for NovaCron
 */

const { execSync } = require('child_process');

module.exports = {
  name: 'storage:optimize',
  description: 'Optimize storage with compression and deduplication',
  usage: '/storage:optimize [--vm <vm-id>] [--compress] [--dedupe] [--analyze]',
  
  async execute(args) {
    const vmIndex = args.indexOf('--vm');
    const vmId = vmIndex !== -1 ? args[vmIndex + 1] : null;
    const compress = args.includes('--compress');
    const dedupe = args.includes('--dedupe');
    const analyze = args.includes('--analyze');
    
    // Default to analyze if no specific action
    const action = !compress && !dedupe ? 'analyze' : null;
    
    console.log('💾 Storage Optimization Tool\n');
    
    try {
      // First, analyze storage if requested or as default
      if (analyze || action === 'analyze') {
        console.log('🔍 Analyzing storage usage...');
        
        const endpoint = vmId 
          ? `/api/storage/analysis?vm_id=${vmId}`
          : '/api/storage/analysis';
        
        const analysisResponse = execSync(
          `curl -s http://localhost:8090${endpoint}`,
          { encoding: 'utf8' }
        );
        const analysis = JSON.parse(analysisResponse);
        
        console.log('\n📊 Storage Analysis:');
        console.log(`  Total Size: ${analysis.total_size_gb} GB`);
        console.log(`  Used Space: ${analysis.used_size_gb} GB (${analysis.usage_percent}%)`);
        console.log(`  Duplicates: ${analysis.duplicate_size_gb} GB`);
        console.log(`  Compressible: ${analysis.compressible_size_gb} GB`);
        console.log(`  Potential Savings: ${analysis.potential_savings_gb} GB`);
        
        if (analysis.recommendations) {
          console.log('\n💡 Recommendations:');
          analysis.recommendations.forEach(rec => {
            console.log(`  - ${rec}`);
          });
        }
      }
      
      // Apply compression if requested
      if (compress) {
        console.log('\n🗜️ Applying compression...');
        
        const compressionData = {
          target: vmId || 'all',
          algorithm: 'zstd',
          level: 3
        };
        
        const compressResponse = execSync(
          `curl -X POST -H "Content-Type: application/json" ` +
          `-d '${JSON.stringify(compressionData)}' ` +
          `http://localhost:8090/api/storage/compress`,
          { encoding: 'utf8' }
        );
        const compressResult = JSON.parse(compressResponse);
        
        console.log('✅ Compression completed');
        console.log(`  Original Size: ${compressResult.original_size_gb} GB`);
        console.log(`  Compressed Size: ${compressResult.compressed_size_gb} GB`);
        console.log(`  Space Saved: ${compressResult.saved_gb} GB (${compressResult.compression_ratio}%)`);
      }
      
      // Apply deduplication if requested
      if (dedupe) {
        console.log('\n🔄 Applying deduplication...');
        
        const dedupeData = {
          target: vmId || 'all',
          method: 'content-hash'
        };
        
        const dedupeResponse = execSync(
          `curl -X POST -H "Content-Type: application/json" ` +
          `-d '${JSON.stringify(dedupeData)}' ` +
          `http://localhost:8090/api/storage/deduplicate`,
          { encoding: 'utf8' }
        );
        const dedupeResult = JSON.parse(dedupeResponse);
        
        console.log('✅ Deduplication completed');
        console.log(`  Blocks Analyzed: ${dedupeResult.blocks_analyzed}`);
        console.log(`  Duplicates Found: ${dedupeResult.duplicates_found}`);
        console.log(`  Space Saved: ${dedupeResult.saved_gb} GB`);
        console.log(`  Dedup Ratio: ${dedupeResult.dedup_ratio}:1`);
      }
      
      // Final storage status
      console.log('\n📈 Final Storage Status:');
      const statusResponse = execSync(
        'curl -s http://localhost:8090/api/storage/status',
        { encoding: 'utf8' }
      );
      const status = JSON.parse(statusResponse);
      
      console.log(`  Total Capacity: ${status.total_capacity_gb} GB`);
      console.log(`  Used Space: ${status.used_gb} GB`);
      console.log(`  Free Space: ${status.free_gb} GB`);
      console.log(`  Compression Enabled: ${status.compression_enabled ? 'Yes' : 'No'}`);
      console.log(`  Dedup Enabled: ${status.dedup_enabled ? 'Yes' : 'No'}`);
      
      return {
        success: true,
        optimizations_applied: [
          compress && 'compression',
          dedupe && 'deduplication'
        ].filter(Boolean)
      };
      
    } catch (error) {
      console.error('❌ Storage optimization failed:', error.message);
      return {
        error: error.message
      };
    }
  }
};