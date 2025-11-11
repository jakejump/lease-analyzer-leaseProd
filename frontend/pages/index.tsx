import Head from 'next/head';
import Link from 'next/link';
import LeaseQA from '../components/LeaseQA';

export default function Home() {
  return (
    <>
      <Head>
        <title>Lease Analyzer</title>
      </Head>
      <main className="min-h-screen bg-gray-200 p-8">
        <div className="max-w-6xl mx-auto bg-gray-900 rounded-lg shadow p-6 space-y-4">
          <div className="flex items-center justify-between">
            <h1 className="text-2xl font-bold">Lease.AI</h1>
            <Link href="/projects" className="bg-blue-600 text-white px-3 py-2 rounded">Projects</Link>
          </div>
          <LeaseQA />
        </div>
      </main>
    </>
  );
}
