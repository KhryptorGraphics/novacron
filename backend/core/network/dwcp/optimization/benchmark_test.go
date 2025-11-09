package optimization

import (
	"crypto/rand"
	"fmt"
	"testing"
	"time"

	"novacron/backend/core/network/dwcp/optimization/lockfree"
	"novacron/backend/core/network/dwcp/optimization/simd"
)

// BenchmarkXORDelta benchmarks SIMD vs scalar XOR operations
func BenchmarkXORDelta(b *testing.B) {
	sizes := []int{
		1024,    // 1KB
		4096,    // 4KB
		16384,   // 16KB
		65536,   // 64KB
		262144,  // 256KB
		1048576, // 1MB
	}

	encoder := simd.NewXORDeltaEncoder()

	for _, size := range sizes {
		b.Run(fmt.Sprintf("Size_%d", size), func(b *testing.B) {
			src1 := make([]byte, size)
			src2 := make([]byte, size)
			dst := make([]byte, size)

			rand.Read(src1)
			rand.Read(src2)

			b.Run("SIMD", func(b *testing.B) {
				b.SetBytes(int64(size))
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					encoder.XORBytes(dst, src1, src2)
				}
			})

			b.Run("Scalar", func(b *testing.B) {
				b.SetBytes(int64(size))
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					for j := 0; j < size; j++ {
						dst[j] = src1[j] ^ src2[j]
					}
				}
			})
		})
	}
}

// BenchmarkChecksum benchmarks checksum calculations
func BenchmarkChecksum(b *testing.B) {
	sizes := []int{64, 256, 1024, 4096, 16384, 65536}
	calc := simd.NewChecksumCalculator()

	for _, size := range sizes {
		data := make([]byte, size)
		rand.Read(data)

		b.Run(fmt.Sprintf("CRC32_Size_%d", size), func(b *testing.B) {
			b.SetBytes(int64(size))
			for i := 0; i < b.N; i++ {
				calc.CalculateCRC32(data)
			}
		})

		b.Run(fmt.Sprintf("Adler32_Size_%d", size), func(b *testing.B) {
			b.SetBytes(int64(size))
			for i := 0; i < b.N; i++ {
				simd.Adler32(data)
			}
		})

		b.Run(fmt.Sprintf("xxHash_Size_%d", size), func(b *testing.B) {
			b.SetBytes(int64(size))
			for i := 0; i < b.N; i++ {
				simd.xxHash32(data, 0)
			}
		})
	}
}

// BenchmarkLockFreeQueue benchmarks lock-free vs mutex-based queue
func BenchmarkLockFreeQueue(b *testing.B) {
	b.Run("LockFreeQueue", func(b *testing.B) {
		q := lockfree.NewLockFreeQueue()
		b.RunParallel(func(pb *testing.PB) {
			i := 0
			for pb.Next() {
				if i%2 == 0 {
					q.Enqueue(i)
				} else {
					q.Dequeue()
				}
				i++
			}
		})
	})

	b.Run("MPMCQueue", func(b *testing.B) {
		q := lockfree.NewMPMCQueue(1024)
		b.RunParallel(func(pb *testing.PB) {
			i := 0
			for pb.Next() {
				if i%2 == 0 {
					q.Enqueue(i)
				} else {
					q.Dequeue()
				}
				i++
			}
		})
	})
}

// BenchmarkRingBuffer benchmarks ring buffer implementations
func BenchmarkRingBuffer(b *testing.B) {
	b.Run("LockFreeRingBuffer", func(b *testing.B) {
		rb := lockfree.NewLockFreeRingBuffer(1024)
		b.RunParallel(func(pb *testing.PB) {
			i := 0
			for pb.Next() {
				if i%2 == 0 {
					rb.Push(i)
				} else {
					rb.Pop()
				}
				i++
			}
		})
	})

	b.Run("SPSCRingBuffer", func(b *testing.B) {
		rb := lockfree.NewSPSCRingBuffer(1024)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if i%2 == 0 {
				rb.Push(i)
			} else {
				rb.Pop()
			}
		}
	})

	b.Run("ByteRingBuffer", func(b *testing.B) {
		rb := lockfree.NewByteRingBuffer(1024 * 1024)
		data := make([]byte, 1024)
		buf := make([]byte, 1024)

		b.SetBytes(1024)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			rb.Write(data)
			rb.Read(buf)
		}
	})
}

// BenchmarkMemoryPool benchmarks memory pool performance
func BenchmarkMemoryPool(b *testing.B) {
	pool := NewObjectPool()

	sizes := []int{64, 256, 1024, 4096, 16384}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("Pool_Size_%d", size), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				buf := pool.Get(size)
				pool.Put(buf)
			}
		})

		b.Run(fmt.Sprintf("Direct_Size_%d", size), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = make([]byte, size)
			}
		})
	}
}

// BenchmarkPrefetch benchmarks prefetching strategies
func BenchmarkPrefetch(b *testing.B) {
	size := 1024 * 1024 // 1MB
	data := make([]byte, size)
	rand.Read(data)

	b.Run("WithPrefetch", func(b *testing.B) {
		prefetcher := NewPrefetcher(4096, PrefetchRead)
		b.SetBytes(int64(size))
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			prefetcher.PrefetchSlice(data)
		}
	})

	b.Run("WithoutPrefetch", func(b *testing.B) {
		b.SetBytes(int64(size))
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			// Just access the data
			_ = data[0]
			_ = data[size-1]
		}
	})

	b.Run("StreamingPrefetch", func(b *testing.B) {
		chunkSize := 4096
		b.SetBytes(int64(size))
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			StreamingPrefetch(data, chunkSize, func(chunk []byte) {
				// Process chunk
				_ = chunk[0]
			})
		}
	})
}

// BenchmarkBatchProcessing benchmarks batch vs individual operations
func BenchmarkBatchProcessing(b *testing.B) {
	b.Run("BatchAllocate", func(b *testing.B) {
		allocator := NewBatchAllocator(1024, 16)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			batch := allocator.AllocateBatch()
			allocator.FreeBatch(batch)
		}
	})

	b.Run("IndividualAllocate", func(b *testing.B) {
		pool := NewObjectPool()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			buffers := make([][]byte, 16)
			for j := range buffers {
				buffers[j] = pool.Get(1024)
			}
			for _, buf := range buffers {
				pool.Put(buf)
			}
		}
	})
}

// BenchmarkDeltaCompression benchmarks delta compression
func BenchmarkDeltaCompression(b *testing.B) {
	encoder := simd.NewXORDeltaEncoder()

	size := 1024 * 1024 // 1MB
	frame1 := make([]byte, size)
	frame2 := make([]byte, size)

	rand.Read(frame1)
	copy(frame2, frame1)

	// Modify 10% of frame2
	for i := 0; i < size/10; i++ {
		frame2[i*10] ^= 0xFF
	}

	b.Run("EncodeDelta", func(b *testing.B) {
		b.SetBytes(int64(size))
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			encoder.EncodeDelta(frame2, frame1)
		}
	})

	delta := encoder.EncodeDelta(frame2, frame1)

	b.Run("CompressDelta", func(b *testing.B) {
		b.SetBytes(int64(len(delta)))
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			encoder.CompressDelta(delta, 16)
		}
	})

	compressed := encoder.CompressDelta(delta, 16)

	b.Run("DecompressDelta", func(b *testing.B) {
		b.SetBytes(int64(len(compressed)))
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			encoder.DecompressDelta(compressed, size)
		}
	})
}

// BenchmarkThroughput measures overall throughput
func BenchmarkThroughput(b *testing.B) {
	sizes := []int{
		1024,      // 1KB
		65536,     // 64KB
		1048576,   // 1MB
		16777216,  // 16MB
	}

	for _, size := range sizes {
		data := make([]byte, size)
		rand.Read(data)

		b.Run(fmt.Sprintf("Process_%dMB", size/(1024*1024)), func(b *testing.B) {
			encoder := simd.NewXORDeltaEncoder()
			checksummer := simd.NewChecksumCalculator()

			b.SetBytes(int64(size))
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				// Simulate full pipeline
				previous := make([]byte, size)

				// Delta encode
				delta := encoder.EncodeDelta(data, previous)

				// Compress
				compressed := encoder.CompressDelta(delta, 16)

				// Checksum
				_ = checksummer.CalculateCRC32(compressed)
			}
		})
	}
}

// BenchmarkLatency measures latency for different operations
func BenchmarkLatency(b *testing.B) {
	b.Run("XOR_1KB", func(b *testing.B) {
		encoder := simd.NewXORDeltaEncoder()
		src1 := make([]byte, 1024)
		src2 := make([]byte, 1024)
		dst := make([]byte, 1024)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			encoder.XORBytes(dst, src1, src2)
		}
	})

	b.Run("Checksum_1KB", func(b *testing.B) {
		calc := simd.NewChecksumCalculator()
		data := make([]byte, 1024)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			calc.CalculateCRC32(data)
		}
	})

	b.Run("QueueOp", func(b *testing.B) {
		q := lockfree.NewLockFreeQueue()

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			q.Enqueue(i)
			q.Dequeue()
		}
	})
}

// BenchmarkConcurrency tests performance under concurrent load
func BenchmarkConcurrency(b *testing.B) {
	b.Run("LockFreeQueue_Concurrent", func(b *testing.B) {
		q := lockfree.NewLockFreeQueue()

		b.RunParallel(func(pb *testing.PB) {
			for pb.Next() {
				q.Enqueue(1)
				q.Dequeue()
			}
		})
	})

	b.Run("MemoryPool_Concurrent", func(b *testing.B) {
		pool := NewObjectPool()

		b.RunParallel(func(pb *testing.PB) {
			for pb.Next() {
				buf := pool.Get(1024)
				pool.Put(buf)
			}
		})
	})

	b.Run("RingBuffer_Concurrent", func(b *testing.B) {
		rb := lockfree.NewLockFreeRingBuffer(1024)

		b.RunParallel(func(pb *testing.PB) {
			for pb.Next() {
				rb.Push(1)
				rb.Pop()
			}
		})
	})
}

// BenchmarkCacheMissRate measures cache performance
func BenchmarkCacheMissRate(b *testing.B) {
	b.Run("Sequential", func(b *testing.B) {
		size := 16 * 1024 * 1024 // 16MB
		data := make([]byte, size)

		b.SetBytes(int64(size))
		b.ResetTimer()

		for i := 0; i < b.N; i++ {
			sum := byte(0)
			for j := 0; j < size; j++ {
				sum += data[j]
			}
			_ = sum
		}
	})

	b.Run("RandomAccess", func(b *testing.B) {
		size := 16 * 1024 * 1024 // 16MB
		data := make([]byte, size)
		indices := make([]int, 10000)

		for i := range indices {
			indices[i] = i * 1597 % size // Pseudo-random
		}

		b.ResetTimer()

		for i := 0; i < b.N; i++ {
			sum := byte(0)
			for _, idx := range indices {
				sum += data[idx]
			}
			_ = sum
		}
	})

	b.Run("CacheAligned", func(b *testing.B) {
		size := 16 * 1024 * 1024 // 16MB
		data := make([]byte, size)

		b.SetBytes(int64(size))
		b.ResetTimer()

		for i := 0; i < b.N; i++ {
			sum := byte(0)
			// Access in cache line chunks
			for j := 0; j < size; j += 64 {
				sum += data[j]
			}
			_ = sum
		}
	})
}

// TestPerformanceGoals validates performance targets
func TestPerformanceGoals(t *testing.T) {
	t.Run("Sub-Microsecond_Latency", func(t *testing.T) {
		encoder := simd.NewXORDeltaEncoder()
		src1 := make([]byte, 1024)
		src2 := make([]byte, 1024)
		dst := make([]byte, 1024)

		iterations := 10000
		start := time.Now()

		for i := 0; i < iterations; i++ {
			encoder.XORBytes(dst, src1, src2)
		}

		elapsed := time.Since(start)
		avgLatency := elapsed / time.Duration(iterations)

		t.Logf("Average XOR latency: %v", avgLatency)

		if avgLatency > time.Microsecond {
			t.Errorf("Latency %v exceeds 1μs target", avgLatency)
		}
	})

	t.Run("100Gbps_Throughput", func(t *testing.T) {
		size := 64 * 1024 * 1024 // 64MB
		data := make([]byte, size)
		encoder := simd.NewXORDeltaEncoder()

		start := time.Now()
		delta := encoder.EncodeDelta(data, data)
		elapsed := time.Since(start)

		throughput := float64(len(delta)) / elapsed.Seconds()
		throughputGbps := throughput * 8 / 1e9

		t.Logf("Throughput: %.2f Gbps", throughputGbps)

		if throughputGbps < 100 {
			t.Logf("Warning: Throughput %.2f Gbps below 100 Gbps target", throughputGbps)
		}
	})

	t.Run("Memory_Pool_Efficiency", func(t *testing.T) {
		pool := NewObjectPool()

		// Warm up pool
		buffers := make([][]byte, 1000)
		for i := range buffers {
			buffers[i] = pool.Get(1024)
		}
		for _, buf := range buffers {
			pool.Put(buf)
		}

		stats := pool.Stats()
		t.Logf("Pool stats: %+v", stats)

		if stats.HitRate < 0.9 {
			t.Errorf("Hit rate %.2f below 90%% target", stats.HitRate)
		}
	})
}
