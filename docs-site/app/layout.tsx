import type { Metadata } from "next";
import { headers } from "next/headers";
import "./globals.css";

export async function generateMetadata(): Promise<Metadata> {
  const h = await headers();
  const host = h.get("x-forwarded-host") ?? h.get("host") ?? "localhost:3000";
  const proto = h.get("x-forwarded-proto") ?? (host.startsWith("localhost") ? "http" : "https");
  const base = new URL(`${proto}://${host}`);
  return {
    metadataBase: base,
    title: "ChemistryKernels.jl — Full primordial chemistry at accelerator scale",
    description: "The fast, accurate full-network production path for primordial chemistry, with methods, validation, rates, cooling, recombination physics, and usage.",
    keywords: ["primordial chemistry", "astrophysics", "Julia", "GPU", "recombination", "HyRec", "RECFAST"],
    openGraph: {
      title: "ChemistryKernels.jl",
      description: "The full primordial network from recombination to first-star collapse — fast, accurate chemistry on CPU and GPU.",
      type: "website",
      images: [{ url: new URL("/og.png", base).toString(), width: 1200, height: 630, alt: "ChemistryKernels.jl methods overview" }],
    },
    twitter: { card: "summary_large_image", title: "ChemistryKernels.jl", description: "The full primordial chemistry network at accelerator scale.", images: [new URL("/og.png", base).toString()] },
  };
}

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
