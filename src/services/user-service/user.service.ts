import { injectable, inject } from 'inversify';
import { UserRepository } from './user.repository';
import { EventBus } from '../shared/event-bus';
import { CacheService } from '../shared/cache.service';
import { User, UserProfile, UserPreferences, CreateUserDto, UpdateUserDto } from './types';
import { ValidationError, NotFoundError } from '../shared/errors';
import { Logger } from '../shared/logger';

@injectable()
export class UserService {
  constructor(
    @inject('UserRepository') private userRepository: UserRepository,
    @inject('EventBus') private eventBus: EventBus,
    @inject('CacheService') private cacheService: CacheService,
    @inject('Logger') private logger: Logger
  ) {}

  async createUser(createUserDto: CreateUserDto): Promise<User> {
    try {
      this.logger.info('Creating new user', { email: createUserDto.email });
      
      // Validate input
      await this.validateCreateUserDto(createUserDto);
      
      // Check if user already exists
      const existingUser = await this.userRepository.findByEmail(createUserDto.email);
      if (existingUser) {
        throw new ValidationError('User already exists with this email');
      }
      
      // Create user
      const user = await this.userRepository.create(createUserDto);
      
      // Cache user data
      await this.cacheService.set(`user:${user.id}`, user, 3600);
      
      // Emit user created event
      await this.eventBus.publish('user.created', {
        userId: user.id,
        email: user.email,
        timestamp: new Date()
      });
      
      this.logger.info('User created successfully', { userId: user.id });
      return user;
    } catch (error) {
      this.logger.error('Failed to create user', { error: error.message, dto: createUserDto });
      throw error;
    }
  }

  async getUserById(userId: string): Promise<User> {
    try {
      // Try cache first
      const cachedUser = await this.cacheService.get(`user:${userId}`);
      if (cachedUser) {
        return cachedUser as User;
      }
      
      // Fetch from database
      const user = await this.userRepository.findById(userId);
      if (!user) {
        throw new NotFoundError('User not found');
      }
      
      // Cache for future requests
      await this.cacheService.set(`user:${userId}`, user, 3600);
      
      return user;
    } catch (error) {
      this.logger.error('Failed to get user', { userId, error: error.message });
      throw error;
    }
  }

  async updateUser(userId: string, updateUserDto: UpdateUserDto): Promise<User> {
    try {
      this.logger.info('Updating user', { userId });
      
      const user = await this.getUserById(userId);
      const updatedUser = await this.userRepository.update(userId, updateUserDto);
      
      // Invalidate cache
      await this.cacheService.delete(`user:${userId}`);
      
      // Cache updated user
      await this.cacheService.set(`user:${userId}`, updatedUser, 3600);
      
      // Emit user updated event
      await this.eventBus.publish('user.updated', {
        userId,
        changes: updateUserDto,
        timestamp: new Date()
      });
      
      this.logger.info('User updated successfully', { userId });
      return updatedUser;
    } catch (error) {
      this.logger.error('Failed to update user', { userId, error: error.message });
      throw error;
    }
  }

  async getUserProfile(userId: string): Promise<UserProfile> {
    try {
      const cacheKey = `profile:${userId}`;
      const cachedProfile = await this.cacheService.get(cacheKey);
      
      if (cachedProfile) {
        return cachedProfile as UserProfile;
      }
      
      const profile = await this.userRepository.getProfile(userId);
      if (!profile) {
        throw new NotFoundError('User profile not found');
      }
      
      await this.cacheService.set(cacheKey, profile, 1800);
      return profile;
    } catch (error) {
      this.logger.error('Failed to get user profile', { userId, error: error.message });
      throw error;
    }
  }

  async updatePreferences(userId: string, preferences: UserPreferences): Promise<void> {
    try {
      await this.userRepository.updatePreferences(userId, preferences);
      
      // Invalidate related caches
      await this.cacheService.delete(`user:${userId}`);
      await this.cacheService.delete(`profile:${userId}`);
      await this.cacheService.delete(`preferences:${userId}`);
      
      // Emit preferences updated event for matching service
      await this.eventBus.publish('user.preferences.updated', {
        userId,
        preferences,
        timestamp: new Date()
      });
      
      this.logger.info('User preferences updated', { userId });
    } catch (error) {
      this.logger.error('Failed to update preferences', { userId, error: error.message });
      throw error;
    }
  }

  async deactivateUser(userId: string): Promise<void> {
    try {
      await this.userRepository.deactivate(userId);
      
      // Clear all user-related cache
      await this.clearUserCache(userId);
      
      // Emit user deactivated event
      await this.eventBus.publish('user.deactivated', {
        userId,
        timestamp: new Date()
      });
      
      this.logger.info('User deactivated', { userId });
    } catch (error) {
      this.logger.error('Failed to deactivate user', { userId, error: error.message });
      throw error;
    }
  }

  private async validateCreateUserDto(dto: CreateUserDto): Promise<void> {
    if (!dto.email || !dto.password) {
      throw new ValidationError('Email and password are required');
    }
    
    if (dto.age < 18) {
      throw new ValidationError('User must be at least 18 years old');
    }
    
    // Add more validation rules as needed
  }

  private async clearUserCache(userId: string): Promise<void> {
    const cacheKeys = [
      `user:${userId}`,
      `profile:${userId}`,
      `preferences:${userId}`,
      `matches:${userId}`
    ];
    
    await Promise.all(cacheKeys.map(key => this.cacheService.delete(key)));
  }
}