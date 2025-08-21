/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025 Elena Viter
 */

import {User, ChevronDown, UserCircle, LogOut} from "lucide-react";
import {useState} from "react";
import {useAuth} from "react-oidc-context";
import {useNavigate} from "react-router-dom";

export default function UserMenu() {
    const [isDropdownOpen, setIsDropdownOpen] = useState(false);
    const auth = useAuth();
    const navigate = useNavigate();

    const toggleDropdown = () => {
        setIsDropdownOpen(!isDropdownOpen);
    };

    const handleProfileClick = () => {
        setIsDropdownOpen(false);
        navigate("/profile");
    };

    const handleLogoutClick = () => {
        setIsDropdownOpen(false);
        auth.signoutRedirect()
    };

    return (
        <div>
            {/* User Icon Button */}
            <button
                onClick={toggleDropdown}
                className="flex items-center space-x-2 p-2 rounded-lg hover:bg-gray-100 transition-colors"
            >
                <User className="h-6 w-6 text-gray-600"/>
                <ChevronDown className="h-4 w-4 text-gray-600"/>
            </button>

            {/* Dropdown Menu */}
            {isDropdownOpen && (
                <div
                    className="absolute right-0 mt-2 w-48 bg-white rounded-lg shadow-lg border border-gray-200 py-1 z-50">
                    <button
                        onClick={handleProfileClick}
                        className="flex items-center w-full px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 transition-colors"
                    >
                        <UserCircle className="h-4 w-4 mr-3"/>
                        User Profile
                    </button>
                    <button
                        onClick={handleLogoutClick}
                        className="flex items-center w-full px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 transition-colors"
                    >
                        <LogOut className="h-4 w-4 mr-3"/>
                        Logout
                    </button>
                </div>
            )}
        </div>
    );
}