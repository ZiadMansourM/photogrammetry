//
//  ContentView.swift
//  Tridify
//
//  Created by Maged Alosali on 26/04/2023.
//

import SwiftUI

struct ContentView: View {
    
    @AppStorage("onBoarding") var onBoarding = true
    
    var body: some View {
        if onBoarding {
            onBoardingView()
        } else {
            HomeView()
        }
        
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
