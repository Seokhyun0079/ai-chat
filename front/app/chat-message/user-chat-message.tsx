"use client";

type UserChatMessageProps = {
  message: {
    text: string;
  };
};

export default function UserChatMessage({ message }: UserChatMessageProps) {
  return (
    <div className="flex items-start gap-2.5 justify-end">
      <div className="flex flex-col gap-1 w-full max-w-[320px] leading-1.5 p-4 border-gray-200 bg-blue-500 rounded-s-xl rounded-ee-xl">
        <p className="text-sm font-normal text-white">{message.text}</p>
      </div>
    </div>
  );
}
