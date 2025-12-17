import React, { useState, useEffect } from 'react';
import { healthService } from '../services/healthService';

const ApiStatusDisplay = () => {
  const [status, setStatus] = useState('checking'); // 'checking', 'healthy', 'unhealthy'
  const [message, setMessage] = useState('Checking backend connection...');
  const [lastChecked, setLastChecked] = useState(null);

  useEffect(() => {
    checkHealth();
    // Set up periodic health checks
    const interval = setInterval(checkHealth, 30000); // Check every 30 seconds

    return () => clearInterval(interval);
  }, []);

  const checkHealth = async () => {
    try {
      setStatus('checking');
      setMessage('Checking backend connection...');

      const healthData = await healthService.checkHealth();

      setStatus(healthData.status);
      setMessage(healthData.message || 'Backend is operational');
      setLastChecked(new Date().toLocaleTimeString());
    } catch (error) {
      setStatus('unhealthy');
      setMessage(error.message || 'Backend is unavailable');
      setLastChecked(new Date().toLocaleTimeString());
    }
  };

  const getStatusColor = () => {
    switch (status) {
      case 'healthy':
        return 'text-green-600 bg-green-100';
      case 'unhealthy':
        return 'text-red-600 bg-red-100';
      case 'checking':
        return 'text-yellow-600 bg-yellow-100';
      default:
        return 'text-gray-600 bg-gray-100';
    }
  };

  const getStatusIcon = () => {
    switch (status) {
      case 'healthy':
        return '✅';
      case 'unhealthy':
        return '❌';
      case 'checking':
        return '⏳';
      default:
        return '❓';
    }
  };

  return (
    <div className="api-status-display p-4 border rounded-lg shadow-sm">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <span className="text-2xl">{getStatusIcon()}</span>
          <div>
            <h3 className="font-semibold text-gray-800">Backend Status</h3>
            <p className="text-sm text-gray-600">{message}</p>
            {lastChecked && (
              <p className="text-xs text-gray-500">Last checked: {lastChecked}</p>
            )}
          </div>
        </div>
        <button
          onClick={checkHealth}
          disabled={status === 'checking'}
          className={`px-3 py-1 rounded text-sm ${
            status === 'checking'
              ? 'bg-gray-200 text-gray-500 cursor-not-allowed'
              : 'bg-blue-500 text-white hover:bg-blue-600'
          }`}
        >
          {status === 'checking' ? 'Checking...' : 'Refresh'}
        </button>
      </div>
      <div className="mt-2">
        <span
          className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor()}`}
        >
          {status.charAt(0).toUpperCase() + status.slice(1)}
        </span>
      </div>
    </div>
  );
};

export default ApiStatusDisplay;