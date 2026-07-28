import "./globals.css";

export const metadata = {
  title: "NsysScope · GPU 性能分析",
  description: "Interactive Nsight Systems performance analysis dashboard",
};

export default function RootLayout({ children }) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
