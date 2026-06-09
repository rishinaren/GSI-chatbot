import { useEffect, useState } from "react";
import ChatApp from "./ChatApp";
import LandingPage from "./components/LandingPage";

function usePathname() {
  const [pathname, setPathname] = useState(() => window.location.pathname);

  useEffect(() => {
    const onNavigate = () => setPathname(window.location.pathname);
    window.addEventListener("popstate", onNavigate);
    return () => window.removeEventListener("popstate", onNavigate);
  }, []);

  return pathname;
}

export default function App() {
  const pathname = usePathname();
  const isAppRoute = pathname === "/app" || pathname.startsWith("/app/");

  if (isAppRoute) {
    return <ChatApp />;
  }

  return <LandingPage />;
}
