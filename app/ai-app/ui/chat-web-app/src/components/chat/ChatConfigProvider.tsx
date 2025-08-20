import { useCallback, useEffect, useState } from 'react';
import {getKBAPIBaseAddress, getCustomEmbedingEndpoint} from "../../AppConfig.ts";

/**
 * @Deprocated we must remove this abomination
 * Configuration interface defining all application settings
 */
export interface AppConfig {
    // API Keys
    openai_api_key: string;
    claude_api_key: string;

    // Custom Model Configuration
    custom_model_endpoint: string;
    custom_model_api_key: string;
    custom_model_name: string;

    // Embedding Configuration
    selected_embedder: string;
    custom_embedding_endpoint: string;
    kb_search_endpoint: string;

    // Model Selection
    selected_model: string;

    // UI Preferences
    show_steps: boolean;
    show_config: boolean;

    // Advanced Settings
    max_retries: number;
    timeout: number;
    debug_mode: boolean;
}

/**
 * Configuration provider options
 */
export interface ConfigProviderOptions {
    storageKey?: string;
    encryptionKey?: string;
}

/**
 * Event types for configuration changes
 */
export type ConfigEvent =
    | 'loaded'
    | 'saved'
    | 'changed'
    | 'updated'
    | 'reset'
    | 'cleared'
    | 'imported'
    | 'error';

/**
 * Event data for configuration changes
 */
export interface ConfigEventData {
    type?: string;
    error?: Error;
    key?: keyof AppConfig;
    oldValue?: any;
    newValue?: any;
    changes?: Array<{
        key: keyof AppConfig;
        oldValue: any;
        newValue: any;
    }>;
    oldConfig?: AppConfig;
    merge?: boolean;
    version?: string;
}

/**
 * Configuration listener function type
 */
export type ConfigListener = (
    event: ConfigEvent,
    data: ConfigEventData,
    currentConfig: AppConfig
) => void;

/**
 * Export data structure
 */
export interface ConfigExportData {
    exported_at: string;
    version: string;
    config: Partial<AppConfig>;
}

/**
 * Validation result interface
 */
export interface ValidationResult {
    isValid: boolean;
    errors: string[];
    hasValidModelKey: boolean;
    hasValidEmbedding: boolean;
}

/**
 * Debug information interface
 */
export interface DebugInfo {
    storageKey: string;
    hasLocalStorage: boolean;
    storageSize: number;
    listenerCount: number;
    isValid: boolean;
    validationErrors: string[];
    hasChanges: boolean;
    config: AppConfig;
}

/**
 * ConfigProvider - Manages application configuration with localStorage persistence
 * Handles storing, loading, and validating user configuration settings
 */
export class ConfigProvider {
    private readonly storageKey: string;
    private readonly encryptionKey: string;
    private readonly listeners: Set<ConfigListener>;
    private readonly defaultConfig: AppConfig;
    private config: AppConfig;

    constructor(options: ConfigProviderOptions = {}) {
        this.storageKey = options.storageKey || 'ai_assistant_config';
        this.encryptionKey = options.encryptionKey || 'ai_config_key';
        this.listeners = new Set();

        // Default configuration
        this.defaultConfig = {
            openai_api_key: '',
            claude_api_key: '',
            custom_model_endpoint: '',
            custom_model_api_key: '',
            custom_model_name: '',
            selected_embedder: 'openai-text-embedding-3-small',
            custom_embedding_endpoint: getCustomEmbedingEndpoint(),
            kb_search_endpoint: `${getKBAPIBaseAddress()}/api/kb/search`,
            selected_model: 'gpt-4o',

            // UI preferences
            show_steps: true,
            show_config: false,

            // Advanced settings
            max_retries: 3,
            timeout: 30000,
            debug_mode: false
        };

        // Current configuration state
        this.config = { ...this.defaultConfig };

        // Load from localStorage on initialization
        this.loadConfig();
    }

    /**
     * Simple encryption/decryption for API keys (basic obfuscation)
     * Note: This is NOT secure encryption, just basic obfuscation
     */
    private encrypt(text: string): string {
        if (!text) return '';
        return btoa(encodeURIComponent(text));
    }

    private decrypt(encryptedText: string): string {
        if (!encryptedText) return '';
        try {
            return decodeURIComponent(atob(encryptedText));
        } catch (error) {
            console.warn('Failed to decrypt value:', error);
            return '';
        }
    }

    /**
     * Prepare config for storage (encrypt sensitive fields)
     */
    private prepareForStorage(config: AppConfig): Record<string, any> {
        const sensitiveFields: (keyof AppConfig)[] = [
            'openai_api_key',
            'claude_api_key',
            'custom_model_api_key'
        ];
        const prepared = { ...config } as Record<string, any>;

        sensitiveFields.forEach(field => {
            if (prepared[field]) {
                prepared[field] = this.encrypt(prepared[field]);
            }
        });

        return prepared;
    }

    /**
     * Prepare config after loading (decrypt sensitive fields)
     */
    private prepareAfterLoading(config: Record<string, any>): AppConfig {
        const sensitiveFields: (keyof AppConfig)[] = [
            'openai_api_key',
            'claude_api_key',
            'custom_model_api_key'
        ];
        const prepared = { ...config } as Record<string, any>;

        sensitiveFields.forEach(field => {
            if (prepared[field]) {
                prepared[field] = this.decrypt(prepared[field]);
            }
        });

        return prepared as AppConfig;
    }

    /**
     * Load configuration from localStorage
     */
    public loadConfig(): void {
        try {
            const stored = localStorage.getItem(this.storageKey);
            if (stored) {
                const parsed = JSON.parse(stored);
                const decrypted = this.prepareAfterLoading(parsed);

                // Merge with defaults to handle new fields
                this.config = {
                    ...this.defaultConfig,
                    ...decrypted
                };

                console.log('Configuration loaded from localStorage');
                this.notifyListeners('loaded', {});
            } else {
                console.log('No stored configuration found, using defaults');
            }
        } catch (error) {
            console.error('Failed to load configuration from localStorage:', error);
            this.config = { ...this.defaultConfig };
        }
    }

    /**
     * Save configuration to localStorage
     */
    public saveConfig(): boolean {
        try {
            const encrypted = this.prepareForStorage(this.config);
            localStorage.setItem(this.storageKey, JSON.stringify(encrypted));
            console.log('Configuration saved to localStorage');
            this.notifyListeners('saved', {});
            return true;
        } catch (error) {
            console.error('Failed to save configuration to localStorage:', error);
            this.notifyListeners('error', { type: 'save', error: error as Error });
            return false;
        }
    }

    /**
     * Get current configuration
     */
    public getConfig(): AppConfig {
        return { ...this.config };
    }

    /**
     * Get a specific configuration value
     */
    public get<K extends keyof AppConfig>(key: K): AppConfig[K] {
        return this.config[key];
    }

    /**
     * Set a configuration value
     */
    public set<K extends keyof AppConfig>(key: K, value: AppConfig[K]): this {
        const oldValue = this.config[key];
        this.config[key] = value;

        // Auto-save after each change
        this.saveConfig();

        this.notifyListeners('changed', { key, oldValue, newValue: value });
        return this;
    }

    /**
     * Update multiple configuration values
     */
    public update(updates: Partial<AppConfig>): this {
        const changes: Array<{
            key: keyof AppConfig;
            oldValue: any;
            newValue: any;
        }> = [];

        Object.entries(updates).forEach(([key, value]) => {
            const configKey = key as keyof AppConfig;
            const oldValue = this.config[configKey];
            if (oldValue !== value) {
                this.config[configKey] = value as any;
                changes.push({ key: configKey, oldValue, newValue: value });
            }
        });

        if (changes.length > 0) {
            this.saveConfig();
            this.notifyListeners('updated', { changes });
        }

        return this;
    }

    /**
     * Reset configuration to defaults
     */
    public reset(): this {
        const oldConfig = { ...this.config };
        this.config = { ...this.defaultConfig };
        this.saveConfig();
        this.notifyListeners('reset', { oldConfig });
        return this;
    }

    /**
     * Clear configuration from localStorage
     */
    public clear(): boolean {
        try {
            localStorage.removeItem(this.storageKey);
            this.config = { ...this.defaultConfig };
            console.log('Configuration cleared from localStorage');
            this.notifyListeners('cleared', {});
            return true;
        } catch (error) {
            console.error('Failed to clear configuration:', error);
            return false;
        }
    }

    /**
     * Validation methods
     */
    public validate(): ValidationResult {
        const hasValidModelKey = this.hasValidModelKey();
        const hasValidEmbedding = this.hasValidEmbedding();
        const errors = this.getValidationErrors();

        return {
            isValid: hasValidModelKey && hasValidEmbedding,
            errors,
            hasValidModelKey,
            hasValidEmbedding
        };
    }

    public isValid(): boolean {
        return this.hasValidModelKey() && this.hasValidEmbedding();
    }

    public hasValidModelKey(): boolean {
        return !!(this.config.openai_api_key?.trim() || this.config.claude_api_key?.trim());
    }

    public hasValidEmbedding(): boolean {
        const selectedEmbedder = this.config.selected_embedder;

        if (selectedEmbedder?.includes('openai')) {
            return !!this.config.openai_api_key?.trim();
        } else {
            return !!this.config.custom_embedding_endpoint?.trim();
        }
    }

    public getValidationErrors(): string[] {
        const errors: string[] = [];

        if (!this.hasValidModelKey()) {
            errors.push('At least one model API key is required (OpenAI or Claude)');
        }

        if (!this.hasValidEmbedding()) {
            const selectedEmbedder = this.config.selected_embedder;
            if (selectedEmbedder?.includes('openai')) {
                errors.push('OpenAI API key is required for selected embedding model');
            } else {
                errors.push('Custom embedding endpoint is required for selected embedding model');
            }
        }

        return errors;
    }

    /**
     * Export/Import functionality
     */
    public export(includeSecrets = false): ConfigExportData {
        const config = { ...this.config };

        if (!includeSecrets) {
            // Remove sensitive fields
            delete (config as any).openai_api_key;
            delete (config as any).claude_api_key;
            delete (config as any).custom_model_api_key;
        }

        return {
            exported_at: new Date().toISOString(),
            version: '1.0',
            config
        };
    }

    public import(exportedData: ConfigExportData, merge = true): boolean {
        try {
            if (!exportedData.config) {
                throw new Error('Invalid export data format');
            }

            const importedConfig = exportedData.config;

            if (merge) {
                this.update(importedConfig);
            } else {
                this.config = {
                    ...this.defaultConfig,
                    ...importedConfig
                } as AppConfig;
                this.saveConfig();
            }

            this.notifyListeners('imported', { merge, version: exportedData.version });
            return true;
        } catch (error) {
            console.error('Failed to import configuration:', error);
            this.notifyListeners('error', { type: 'import', error: error as Error });
            return false;
        }
    }

    /**
     * Event system for configuration changes
     */
    public subscribe(listener: ConfigListener): () => void {
        this.listeners.add(listener);
        return () => this.listeners.delete(listener);
    }

    public unsubscribe(listener: ConfigListener): boolean {
        return this.listeners.delete(listener);
    }

    private notifyListeners(event: ConfigEvent, data: ConfigEventData): void {
        this.listeners.forEach(listener => {
            try {
                listener(event, data, this.config);
            } catch (error) {
                console.error('Error in config listener:', error);
            }
        });
    }

    /**
     * Utility methods
     */
    public hasChanges(): boolean {
        try {
            const stored = localStorage.getItem(this.storageKey);
            if (!stored) return true;

            const storedConfig = this.prepareAfterLoading(JSON.parse(stored));
            return JSON.stringify(storedConfig) !== JSON.stringify(this.config);
        } catch (error) {
            return true;
        }
    }

    public getStorageSize(): number {
        try {
            const stored = localStorage.getItem(this.storageKey);
            return stored ? new Blob([stored]).size : 0;
        } catch (error) {
            return 0;
        }
    }

    /**
     * Debug information
     */
    public getDebugInfo(): DebugInfo {
        return {
            storageKey: this.storageKey,
            hasLocalStorage: typeof Storage !== 'undefined',
            storageSize: this.getStorageSize(),
            listenerCount: this.listeners.size,
            isValid: this.isValid(),
            validationErrors: this.getValidationErrors(),
            hasChanges: this.hasChanges(),
            config: this.config
        };
    }
}

/**
 * Hook return type
 */
export interface UseConfigProviderReturn {
    config: AppConfig;
    isValid: boolean;
    validationErrors: string[];
    validationResult: ValidationResult;
    updateConfig: (updates: Partial<AppConfig>) => void;
    setConfigValue: <K extends keyof AppConfig>(key: K, value: AppConfig[K]) => void;
    resetConfig: () => void;
    configProvider: ConfigProvider;
}

/**
 * React Hook for using ConfigProvider with TypeScript support
 */
export function useConfigProvider(configProvider: ConfigProvider): UseConfigProviderReturn {
    const [config, setConfig] = useState<AppConfig>(configProvider.getConfig());
    const [validationResult, setValidationResult] = useState<ValidationResult>(
        configProvider.validate()
    );

    useEffect(() => {
        const unsubscribe = configProvider.subscribe((event, data, currentConfig) => {
            setConfig({ ...currentConfig });
            setValidationResult(configProvider.validate());
        });

        return unsubscribe;
    }, [configProvider]);

    const updateConfig = useCallback((updates: Partial<AppConfig>) => {
        configProvider.update(updates);
    }, [configProvider]);

    const setConfigValue = useCallback(<K extends keyof AppConfig>(
        key: K,
        value: AppConfig[K]
    ) => {
        configProvider.set(key, value);
    }, [configProvider]);

    const resetConfig = useCallback(() => {
        configProvider.reset();
    }, [configProvider]);

    return {
        config,
        isValid: validationResult.isValid,
        validationErrors: validationResult.errors,
        validationResult,
        updateConfig,
        setConfigValue,
        resetConfig,
        configProvider
    };
}