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
    title: "ChemistryKernels.jl — Primordial chemistry at accelerator scale",
    description: "Methods, validation, rates, cooling functions, recombination physics, and usage for ChemistryKernels.jl.",
    keywords: ["primordial chemistry", "astrophysics", "Julia", "GPU", "recombination", "HyRec", "RECFAST"],
    openGraph: {
      title: "ChemistryKernels.jl",
      description: "From recombination to the first cooling halos — table-free chemistry on CPU and GPU.",
      type: "website",
      images: [{ url: new URL("/og.png", base).toString(), width: 1200, height: 630, alt: "ChemistryKernels.jl methods overview" }],
    },
    twitter: { card: "summary_large_image", title: "ChemistryKernels.jl", description: "Primordial chemistry at accelerator scale.", images: [new URL("/og.png", base).toString()] },
  };
}

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
