"use client";

import { useState, KeyboardEvent, useRef, useEffect } from "react";

interface SaveChatDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onSave: (aiName: string, userName: string) => void;
}

export default function SaveChatDialog({
  isOpen,
  onClose,
  onSave,
}: SaveChatDialogProps) {
  const [aiName, setAiName] = useState("");
  const [userName, setUserName] = useState("");
  const aiNameInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (isOpen && aiNameInputRef.current) {
      aiNameInputRef.current.focus();
    }
  }, [isOpen]);

  const handleSave = () => {
    if (aiName.trim() && userName.trim()) {
      onSave(aiName.trim(), userName.trim());
      setAiName("");
      setUserName("");
      onClose();
    }
  };

  const handleKeyPress = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      handleSave();
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg p-6 w-96 max-w-[90vw]">
        <h2 className="text-xl font-bold mb-4">대화 기록 저장</h2>

        <div className="space-y-4">
          <div>
            <label
              htmlFor="aiName"
              className="block text-sm font-medium text-gray-700 mb-1"
            >
              AI 이름
            </label>
            <input
              ref={aiNameInputRef}
              type="text"
              id="aiName"
              value={aiName}
              onChange={(e) => setAiName(e.target.value)}
              onKeyPress={handleKeyPress}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="AI 이름을 입력하세요"
            />
          </div>

          <div>
            <label
              htmlFor="userName"
              className="block text-sm font-medium text-gray-700 mb-1"
            >
              내 이름
            </label>
            <input
              type="text"
              id="userName"
              value={userName}
              onChange={(e) => setUserName(e.target.value)}
              onKeyPress={handleKeyPress}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="내 이름을 입력하세요"
            />
          </div>
        </div>

        <div className="flex justify-end space-x-3 mt-6">
          <button
            onClick={onClose}
            className="px-4 py-2 text-gray-600 border border-gray-300 rounded-md hover:bg-gray-50"
          >
            취소
          </button>
          <button
            onClick={handleSave}
            disabled={!aiName.trim() || !userName.trim()}
            className="px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed"
          >
            저장
          </button>
        </div>
      </div>
    </div>
  );
}
