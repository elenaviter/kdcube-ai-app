/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025 Elena Viter
 */

import {useEffect, useState} from "react";
import {Loader, Search} from "lucide-react";
import {FileLoading, FileLoadingError, FilesPreviewProps} from "./Shared.tsx";

const CSVPreview = ({content, loading, error}: FilesPreviewProps) => {
    const [searchTerm, setSearchTerm] = useState('');
    const [parsedData, setParsedData] = useState([]);
    const [headers, setHeaders] = useState<string[]>([]);

    useEffect(() => {
        if (content) {
            try {
                // Basic CSV parsing (for more complex CSVs, we could use Papa Parse)
                const lines = content.split('\n');
                const headers = lines[0].split(',').map(header => header.trim());
                setHeaders(headers);

                const rows = [];
                for (let i = 1; i < lines.length; i++) {
                    if (lines[i].trim() === '') continue;

                    const values = lines[i].split(',');
                    const row = {};
                    headers.forEach((header, index) => {
                        row[header] = values[index] ? values[index].trim() : '';
                    });
                    rows.push(row);
                }
                setParsedData(rows);
            } catch (err) {
                console.error('Error parsing CSV:', err);
            }
        }
    }, [content]);

    const filteredData = parsedData.filter(row =>
        Object.values(row).some(value =>
            value.toString().toLowerCase().includes(searchTerm.toLowerCase())
        )
    );

    if (loading) {
        return <FileLoading />
    }

    if (error) {
        return <FileLoadingError error={error} />;
    }

    return (
        <div className="flex-1 min-h-1 flex flex-col">
            {/* CSV Toolbar */}
            <div className="flex items-center justify-between p-3 bg-gray-100 border-b">
                <div className="flex items-center space-x-2">
                    <div className="relative">
                        <Search size={16} className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400"/>
                        <input
                            type="text"
                            placeholder="Search data..."
                            value={searchTerm}
                            onChange={(e) => setSearchTerm(e.target.value)}
                            className="pl-9 pr-3 py-1 text-sm border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
                        />
                    </div>
                    <span className="text-sm text-gray-600">{filteredData.length} rows</span>
                </div>
            </div>

            {/* CSV Table */}
            <div className="flex-1 overflow-auto">
                {headers.length > 0 ? (
                    <table className="text-sm w-full">
                        <thead className="bg-gray-50 sticky top-0">
                        <tr>
                            {headers.map((header, index) => (
                                <th key={index} className="px-4 py-2 text-left font-medium text-gray-700 border-b">
                                    {header}
                                </th>
                            ))}
                        </tr>
                        </thead>
                        <tbody>
                        {filteredData.map((row, rowIndex) => (
                            <tr key={rowIndex} className={rowIndex % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                                {headers.map((header, cellIndex) => (
                                    <td key={cellIndex} className="px-4 py-2 border-b">
                                        {row[header]}
                                    </td>
                                ))}
                            </tr>
                        ))}
                        </tbody>
                    </table>
                ) : (
                    <div className="flex items-center justify-center h-full text-gray-500">
                        No data available
                    </div>
                )}
            </div>
        </div>
    );
};

export default CSVPreview