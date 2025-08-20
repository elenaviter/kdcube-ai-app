import { User, Mail, LogOut } from "lucide-react";
import {useAuth} from "react-oidc-context";

export default function UserProfile() {
    const auth = useAuth();
    const profile = auth.user?.profile;

    const handleLogout = () => {
        auth.signoutRedirect()
    };

    return (
        <div className="min-h-screen w-full bg-gray-50 p-8">
            <div className="max-w-2xl mx-auto">
                {/* Profile Card */}
                <div className="bg-white rounded-lg shadow-md p-8">
                    {/* Header */}
                    <div className="flex items-center justify-between mb-8">
                        <div className="flex items-center space-x-4">
                            <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center">
                                <User className="h-8 w-8 text-blue-600" />
                            </div>
                            <div>
                                <h1 className="text-2xl font-semibold text-gray-900">{profile?.name}</h1>
                                <p className="text-gray-600">View your account information</p>
                            </div>
                        </div>

                        {/* Logout Button */}
                        <button
                            onClick={handleLogout}
                            className="flex items-center space-x-2 px-6 py-3 bg-red-600 text-white font-medium rounded-lg hover:bg-red-700 transition-colors shadow-md"
                        >
                            <LogOut className="h-5 w-5" />
                            <span>Logout</span>
                        </button>
                    </div>

                    {/* Profile Information */}
                    <div className="space-y-6">
                        {/* Name */}
                        <div className="flex items-center space-x-4">
                            <User className="h-5 w-5 text-gray-500" />
                            <div className="flex-1">
                                <label className="block text-sm font-medium text-gray-700 mb-1">
                                    Name:
                                </label>
                                <p className="text-gray-900">{profile?.given_name}</p>
                            </div>
                        </div>

                        {/* Family name */}
                        <div className="flex items-center space-x-4">
                            <User className="h-5 w-5 text-gray-500" />
                            <div className="flex-1">
                                <label className="block text-sm font-medium text-gray-700 mb-1">
                                    Family name:
                                </label>
                                <p className="text-gray-900">{profile?.family_name}</p>
                            </div>
                        </div>


                        {/* Email */}
                        <div className="flex items-center space-x-4">
                            <Mail className="h-5 w-5 text-gray-500" />
                            <div className="flex-1">
                                <label className="block text-sm font-medium text-gray-700 mb-1">
                                    Email Address
                                </label>
                                <p className="text-gray-900">{profile?.email}</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}